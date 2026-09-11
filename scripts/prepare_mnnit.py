"""Safely extract the MNNIT retail archive and record per-video provenance.

Extraction is path-traversal checked and size capped. The original archive stays
untouched; extracted files keep their archive-relative paths under `_source`, and the
published stems the trainer expects are hardlinks onto them. Scene-level Normal /
Shoplifting folders are recorded as scene-level labels only — they say nothing about
which person in a clip acted, and must not be applied per person at training time.
"""
from pathlib import Path
import hashlib,json,re,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from watchverify.perception import video_info
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'data/raw/mnnit/Dataset.zip'
OUT=ROOT/'data/processed/mnnit'
SOURCE=OUT/'_source'
MANIFEST=ROOT/'data/manifests/mnnit.jsonl'
MAX_EXPANDED=2*1024**3
MAX_MEMBER=256*1024**2
LICENSE='CC BY 4.0 — cite DOI 10.17632/r3yjf35hzr.1'
CLASSES={'Normal':'normal','Shoplifting':'shoplifting'}

def safe_target(member,destination):
    """Reject absolute paths, traversal and symlinks before anything is written."""
    name=member.filename
    if name.startswith('/') or '\\' in name or Path(name).is_absolute():
        raise ValueError(f'Unsafe archive member path: {name!r}')
    target=(destination/name).resolve()
    if not str(target).startswith(str(destination.resolve())+'/'):
        raise ValueError(f'Archive member escapes the destination: {name!r}')
    if (member.external_attr>>16)&0o170000==0o120000:
        raise ValueError(f'Archive member is a symlink: {name!r}')
    return target

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def classify(name):
    parts=Path(name).parts
    for folder,label in CLASSES.items():
        if folder in parts:return label
    return None

def order_key(name):
    match=re.search(r'\((\d+)\)',Path(name).stem)
    return (int(match.group(1)) if match else 0,name)

def main():
    if not ARCHIVE.exists():raise SystemExit(f'Missing archive: {ARCHIVE}')
    OUT.mkdir(parents=True,exist_ok=True);SOURCE.mkdir(parents=True,exist_ok=True)
    rows=[];seen={};counters={label:0 for label in CLASSES.values()}
    skipped=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        members=[m for m in z.infolist() if not m.is_dir() and m.filename.lower().endswith('.mp4')]
        total=sum(m.file_size for m in members)
        if total>MAX_EXPANDED:raise SystemExit(f'Unexpected expansion size: {total} bytes')
        oversized=[m.filename for m in members if m.file_size>MAX_MEMBER]
        if oversized:raise SystemExit(f'Unexpectedly large members: {oversized[:3]}')
        for member in sorted(members,key=lambda m:(classify(m.filename) or '',order_key(m.filename))):
            label=classify(member.filename)
            if label is None:
                skipped.append({'member':member.filename,'reason':'unrecognised_class_folder'});continue
            target=safe_target(member,SOURCE)
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists() or target.stat().st_size!=member.file_size:
                with z.open(member) as src,target.open('wb') as dst:
                    while True:
                        chunk=src.read(1024*1024)
                        if not chunk:break
                        dst.write(chunk)
            sha=digest(target)
            row={'member':member.filename,'sha256':sha,'scene_label':label,
                 'extracted':str(target.relative_to(ROOT)),'bytes':target.stat().st_size,
                 'license':LICENSE,'label_scope':'scene_level_only'}
            if sha in seen:
                row.update(published=None,decodable=None,excluded='duplicate_of_'+seen[sha])
                rows.append(row);continue
            try:
                row['media']=video_info(target);row['decodable']=True
            except Exception as exc:
                row.update(decodable=False,published=None,excluded=f'undecodable: {type(exc).__name__}: {exc}')
                rows.append(row);continue
            counters[label]+=1;stem=f'{label}_{counters[label]:03d}'
            published=OUT/f'{stem}.mp4'
            if published.exists() or published.is_symlink():published.unlink()
            published.hardlink_to(target)
            seen[sha]=stem;row.update(published=stem,excluded=None)
            rows.append(row)
    MANIFEST.parent.mkdir(parents=True,exist_ok=True)
    with MANIFEST.open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    summary={'archive':str(ARCHIVE.relative_to(ROOT)),'archive_sha256':digest(ARCHIVE),
             'members':len(rows),'published':sum(1 for r in rows if r.get('published')),
             'by_class':{k:counters[k] for k in counters},
             'duplicates':[r['member'] for r in rows if (r.get('excluded') or '').startswith('duplicate_of_')],
             'undecodable':[r['member'] for r in rows if r.get('decodable') is False],
             'skipped':skipped,'manifest':str(MANIFEST.relative_to(ROOT))}
    (OUT/'_extraction_report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='archive_sha256'},indent=2))

if __name__=='__main__':main()
