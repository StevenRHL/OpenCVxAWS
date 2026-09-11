"""Convert original RGB PNG archives using supplied camera-0 timestamps, never depth data.

Timing is validated against the publisher's sync table before any encoding, and an
output is published only after it has been decoded back successfully. Unresolvable
timing is reported per sequence; it is never replaced with a plausible substitute.
"""
from pathlib import Path
import csv,json,re,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from watchverify.perception import TIME_BASE,VideoExport,frames,sha256,video_info
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed/urfall'
# Partials keep the .mp4 suffix so the muxer can infer its format, and live in a
# separate directory so an interrupted run can never be globbed as prepared data.
PARTIAL=OUT/'_partial'
PREPARER_VERSION=2
LICENSE='CC BY-NC-SA 4.0 — non-commercial academic research; competition/commercial distribution not cleared'
MAX_EXPANDED=2*1024**3
MAX_MEMBER=64*1024**2

def read_timing(name):
    """Publisher sync table -> {source_frame: seconds}. Raises on unusable timing."""
    path=ROOT/'data/raw/urfall'/f'{name}-data.csv'
    if not path.exists():raise ValueError(f'{name}: missing sync table {path.name}')
    stamps={};previous=None
    for line,row in enumerate(csv.reader(path.open()),start=1):
        if not row:continue
        try:i=int(row[0]);ms=float(row[1])
        except (IndexError,ValueError):raise ValueError(f'{name}: unreadable sync row {line}: {row!r}')
        if i in stamps:raise ValueError(f'{name}: duplicate source frame id {i} at row {line}')
        if previous is not None:
            j,previous_ms=previous
            if i<=j:raise ValueError(f'{name}: source frame ids not increasing ({j} then {i} at row {line})')
            if ms<=previous_ms:raise ValueError(f'{name}: source times not increasing (frame {j} at {previous_ms}ms, frame {i} at {ms}ms)')
        stamps[i]=ms/1000;previous=(i,ms)
    if len(stamps)<2:raise ValueError(f'{name}: sync table has {len(stamps)} usable rows')
    return stamps,path

def check_encodable(name,ordered):
    """Every retained time must land on its own tick at the output timebase."""
    first=ordered[0][1];seen={}
    for i,t in ordered:
        pts=round((t-first)/TIME_BASE)
        if pts in seen:raise ValueError(f'{name}: frames {seen[pts]} and {i} share output tick {pts} at {float(1/TIME_BASE):.0f} Hz')
        seen[pts]=i

def nominal_rate(ordered):
    gaps=np.diff([t for _,t in ordered])
    return float(min(240,max(1,round(1/max(float(np.median(gaps)),1/240)))))

def is_current(name,archive):
    sidecar=OUT/f'{name}.json'
    if not sidecar.exists() or not (OUT/f'{name}.mp4').exists():return False
    try:card=json.loads(sidecar.read_text())
    except json.JSONDecodeError:return False
    return card.get('preparer_version')==PREPARER_VERSION and card.get('archive_sha256')==sha256(archive)

def prepare(archive):
    name=archive.name.replace('-cam0-rgb.zip','')
    if is_current(name,archive):return {'source_id':name,'status':'current'}
    stamps,sync=read_timing(name)
    destination=OUT/f'{name}.mp4';OUT.mkdir(parents=True,exist_ok=True)
    PARTIAL.mkdir(parents=True,exist_ok=True);temporary=PARTIAL/f'{name}.mp4'
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as z:
        members=[m for m in z.infolist() if m.filename.lower().endswith('.png')]
        if not members:raise ValueError(f'{name}: archive contains no PNG frames')
        indexed=[]
        for m in members:
            match=re.search(r'-(\d+)\.png$',m.filename)
            if match is None:raise ValueError(f'{name}: cannot read a frame number from {m.filename!r}')
            if m.file_size>MAX_MEMBER:raise ValueError(f'{name}: {m.filename} is unexpectedly large ({m.file_size} bytes)')
            indexed.append((int(match.group(1)),m))
        indexed.sort(key=lambda v:v[0])
        if sum(m.file_size for _,m in indexed)>MAX_EXPANDED:raise ValueError(f'{name}: unexpected expansion size')
        # Some sequences ship more frames than timing rows (adl-37: 350 frames, 330 rows).
        # A trailing untimed block is discarded and recorded; timing is never invented.
        # An interior gap means the timing table is corrupt, so refuse the sequence.
        missing=[i for i,_ in indexed if i not in stamps]
        dropped=[]
        if missing:
            timed=[i for i,_ in indexed if i in stamps]
            if not timed or set(missing)!={i for i,_ in indexed if i>max(timed)}:
                raise ValueError(f'{name}: {len(missing)} frames have no source timing and they are '
                                 f'not a trailing block (first: {missing[0]}); timing table is unusable')
            dropped=[{'source_frame':i,'reason':'no_published_timing'} for i in sorted(missing)]
            indexed=[(i,m) for i,m in indexed if i in stamps]
        ordered=[(i,stamps[i]) for i,_ in indexed]
        check_encodable(name,ordered)
        first=ordered[0][1];rate=nominal_rate(ordered)
        mapping=[];writer=None
        try:
            for decoded_index,(i,m) in enumerate(indexed):
                frame=cv2.imdecode(np.frombuffer(z.read(m),dtype=np.uint8),cv2.IMREAD_COLOR)
                if frame is None:raise ValueError(f'{name}: frame {i} is not a readable image')
                if writer is None:writer=VideoExport(temporary,frame.shape[1],frame.shape[0],rate)
                t=stamps[i]-first;writer.write(frame,t)
                mapping.append({'source_frame':i,'decoded_index':decoded_index,'output_t':t})
        except BaseException:
            if writer:
                try:writer.close()
                except Exception:pass
            temporary.unlink(missing_ok=True);raise
        writer.close()
    info=video_info(temporary)
    decoded=[(index,t) for index,t,_ in frames(temporary)]
    if len(decoded)!=len(mapping):raise ValueError(f'{name}: encoded {len(mapping)} frames but decoded {len(decoded)}')
    drift=max(abs(t-row['output_t']) for (_,t),row in zip(decoded,mapping))
    if drift>1e-4:raise ValueError(f'{name}: decoded timing drifted by {drift:.6f}s from the source table')
    temporary.replace(destination)
    card={'source_id':name,'preparer_version':PREPARER_VERSION,'archive':str(archive.relative_to(ROOT)),
          'archive_sha256':sha256(archive),'video_sha256':sha256(destination),
          'timestamp_source':str(sync.relative_to(ROOT)),'time_base':str(TIME_BASE),
          'nominal_rate':rate,'frame_count':len(mapping),'max_decode_drift_s':drift,
          'media':info,'dropped':dropped,'frames':mapping,
          'research_only':True,'license':LICENSE}
    (OUT/f'{name}.json').write_text(json.dumps(card,indent=2))
    return {'source_id':name,'status':'prepared','frames':len(mapping),'dropped':len(dropped),'duration_s':round(mapping[-1]['output_t'],3),'nominal_rate':rate,'max_decode_drift_s':drift}

def main():
    archives=sorted((ROOT/'data/raw/urfall').glob('*-cam0-rgb.zip'))
    report={'preparer_version':PREPARER_VERSION,'prepared':[],'current':[],'failed':[]}
    for archive in archives:
        name=archive.name.replace('-cam0-rgb.zip','')
        try:
            result=prepare(archive)
            report['current' if result['status']=='current' else 'prepared'].append(result)
            print(json.dumps(result),flush=True)
        except Exception as exc:
            failure={'source_id':name,'status':'failed','error':f'{type(exc).__name__}: {exc}'}
            report['failed'].append(failure);print(json.dumps(failure),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'_preparation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'archives':len(archives),'prepared':len(report['prepared']),'current':len(report['current']),'failed':len(report['failed'])},indent=2))
    return 1 if report['failed'] else 0

if __name__=='__main__':sys.exit(main())
