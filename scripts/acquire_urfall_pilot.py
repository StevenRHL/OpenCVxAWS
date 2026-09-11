"""Fetch UR Fall camera-0 RGB archives and their sync tables.

The publisher provides 30 fall and 40 ADL sequences. Missing or refused files are
reported per sequence; a gap is never assumed to mean the sequence does not exist.
"""
from pathlib import Path
import argparse,collections,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from acquire_data import fetch
from concurrent.futures import ThreadPoolExecutor

BASE='https://fenix.ur.edu.pl/~mkepski/ds/data/'
LICENSE='CC BY-NC-SA 4.0; non-commercial academic research; not cleared for competition/commercial redistribution'
COUNTS={'fall':30,'adl':40}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--falls',type=int,default=COUNTS['fall'])
    p.add_argument('--adls',type=int,default=COUNTS['adl'])
    a=p.parse_args()
    jobs=[]
    for kind,count in [('fall',min(a.falls,COUNTS['fall'])),('adl',min(a.adls,COUNTS['adl']))]:
        for i in range(1,count+1):
            for ending in ['-cam0-rgb.zip','-data.csv']:
                name=f'{kind}-{i:02}{ending}'
                jobs.append((BASE+name,'data/raw/urfall/'+name,'urfall',LICENSE))
    results=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(lambda x:fetch(*x),jobs):
            results.append(row)
            if row['status'] not in ('cached','downloaded'):
                print(json.dumps({'path':row['path'],'status':row['status'],'error':row.get('error','')[:200]}),flush=True)
    status=collections.Counter(r['status'] for r in results)
    problems=[r for r in results if r['status'] not in ('cached','downloaded')]
    archives=[r for r in results if r['path'].endswith('-cam0-rgb.zip') and r['status'] in ('cached','downloaded')]
    print(json.dumps({'requested':len(jobs),'status':dict(status),
                      'rgb_archives_available':len(archives),
                      'bytes_gib':round(sum(r.get('bytes',0) for r in results)/1024**3,2),
                      'problems':[{'path':r['path'],'status':r['status']} for r in problems]},indent=2))
    return 1 if problems else 0

if __name__=='__main__':sys.exit(main())
