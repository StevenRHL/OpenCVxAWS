"""Bounded, provenance-recorded dataset acquisition; never executes downloaded code."""
from pathlib import Path
import argparse, concurrent.futures, datetime, hashlib, json, shutil, subprocess, threading

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()
MANIFEST = ROOT / 'data/manifests/downloads.jsonl'
LIMIT = 10 * 1024**3
RESERVE = 15 * 1024**3

def record(item):
    with LOCK:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open('a') as f: f.write(json.dumps(item) + '\n')

def fetch(url, relative, source, license_note='Unverified'):
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    row = dict(source_id=source, url=url, path=relative, retrieved_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), license_note=license_note)
    if target.exists():
        row.update(status='cached', bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest()); record(row); return row
    used = sum(p.stat().st_size for p in (ROOT/'data/raw').rglob('*') if p.is_file())
    if used >= LIMIT or shutil.disk_usage(ROOT).free < RESERVE + 256*1024**2:
        row.update(status='blocked_budget'); record(row); return row
    partial = target.with_name(target.name+'.partial')
    command = ['curl','-sS','-L','--fail','--retry','2','--connect-timeout','20','--max-time','240','--max-filesize',str(min(LIMIT-used, 2*1024**3)),'-o',str(partial),'-w','%{url_effective}\n%{http_code}\n%{content_type}',url]
    p = subprocess.run(command, capture_output=True,text=True)
    row['response'] = p.stdout
    if p.returncode:
        row.update(status='failed',error=p.stderr[-1500:]); record(row); return row
    first = partial.open('rb').read(256).lstrip().lower()
    expected_binary = target.suffix.lower() in ('.mp4','.avi','.zip','.npy','.pkl','.mat')
    if expected_binary and (first.startswith(b'<!doctype html') or first.startswith(b'<html')):
        row.update(status='failed_html',error='HTML response instead of requested data'); record(row); return row
    if target.suffix=='.zip' and first[:2]!=b'pk':
        row.update(status='failed_type',error='ZIP magic absent'); record(row); return row
    partial.rename(target)
    row.update(status='downloaded',bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    record(row); print(json.dumps(row),flush=True); return row

def urfall(count=3):
    base='https://fenix.ur.edu.pl/~mkepski/ds/'
    license_note='CC BY-NC-SA 4.0; non-commercial academic research only; no competition/commercial clearance claimed'
    fetch(base+'uf.html','data/metadata/urfall.html','urfall',license_note)
    for name in ['urfall-cam0-falls.csv','urfall-cam0-adls.csv']:
        fetch(base+'data/'+name,'data/raw/urfall/'+name,'urfall',license_note)
    jobs=[]
    for kind, maximum in [('fall',30),('adl',40)]:
        for i in range(1,min(count,maximum)+1):
            for suffix in ['-cam0.mp4','-data.csv']:
                name=f'{kind}-{i:02}{suffix}'
                jobs.append((base+'data/'+name,'data/raw/urfall/'+name,'urfall',license_note))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda j:fetch(*j),jobs))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--urfall-count',type=int); p.add_argument('--url');p.add_argument('--path');p.add_argument('--source');p.add_argument('--license',default='Unverified');a=p.parse_args()
    if a.urfall_count: urfall(a.urfall_count)
    elif a.url: fetch(a.url,a.path,a.source,a.license)
