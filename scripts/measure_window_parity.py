"""Measure what the app emits against what the activity card measured.

The card's numbers come from `windows()` over whole clips. The application emits windows
per person, refuses to summarise across missing time, and counts a run of consecutive
windows. After the parity fix those two agree on every ungapped single-track clip; this
script says exactly where they still do not, so the card's figures can be quoted with a
stated scope instead of an assumption.

Train and validation only. The activity branch has no held-out test report, so there is
nothing here to correct by spending one, and the council's training contract forbids
consuming a held-out split to fill a table.

    .venv/bin/python scripts/measure_window_parity.py
"""
from pathlib import Path
import argparse,datetime,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from watchverify.features import ACTIVITY_SUSTAINED_WINDOWS,aggregator_for
from watchverify.models import Models
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_models import mnnit_eligible,xy

# The cached corpora were extracted at this rate; the app's default matches it.
ANALYSIS_FPS=10
MEASURABLE_SPLITS=('train','validation')


def runtime_windows(source):
    """Windows the application would emit for one clip, in order, with continuity."""
    d=np.load(ROOT/'data/processed/features/mnnit'/f'{source}.npz',allow_pickle=False)
    x=xy(d)
    aggregator=aggregator_for(ANALYSIS_FPS)
    emitted=[]
    for i in range(len(d['t'])):
        for window in aggregator.update(int(d['track'][i]),d['t'][i],x[i][:12],x[i][12:],
                                        d['hip_speed'][i],d['angular_speed'][i],
                                        d['down'][i],d['upright'][i],d['quality'][i]):
            emitted.append(window)
    return emitted


def sustained(flags,continuous=None):
    """Whether a run of consecutive flagged windows ever reaches the alert length.

    `continuous` is the application's extra condition: a window that does not follow the
    previous one starts a new run however soon after it arrives.
    """
    run=0
    for i,flag in enumerate(flags):
        if continuous is not None and not continuous[i]:run=0
        run=run+1 if flag else 0
        if run>=ACTIVITY_SUSTAINED_WINDOWS:return True
    return False


def summarise(rows):
    """Window-level counts and clip-level alert rates for one path."""
    tp=fp=fn=tn=0;alerted={0:0,1:0};clips={0:0,1:0}
    for label,flags,continuous in rows:
        for flag in flags:
            if label==1:tp+=flag;fn+=not flag
            else:fp+=flag;tn+=not flag
        clips[label]+=1
        alerted[label]+=int(sustained(flags,continuous))
    return {'windows':tp+fp+fn+tn,'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn),
            'window_precision':round(tp/max(tp+fp,1),4),'window_recall':round(tp/max(tp+fn,1),4),
            'flagged_normal_window_fraction':round(fp/max(fp+tn,1),4),
            'shoplifting_clips_alerted':f'{alerted[1]}/{clips[1]}',
            'normal_clips_alerted':f'{alerted[0]}/{clips[0]}'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out')
    arguments=parser.parse_args()
    models=Models()
    if 'activity' not in models.loaded:raise SystemExit(f'No usable activity model: {models.status}')
    card=models.loaded['activity'][1]
    eligible=mnnit_eligible()[0]
    report={'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'model':{k:card.get(k) for k in ('run_id','sha256','threshold','window_s','stride_s')},
            'analysis_fps':ANALYSIS_FPS,'sustained_windows':ACTIVITY_SUSTAINED_WINDOWS,
            'splits':{}}
    for split in MEASURABLE_SPLITS:
        rows=[r for r in eligible if r['split']==split]
        offline=[];runtime=[];divergent=[]
        for record in rows:
            offline_flags=[bool(models.score('activity',v)['positive']) for v in record['windows']]
            live=runtime_windows(record['source'])
            live_flags=[bool(models.score('activity',w.descriptor)['positive']) for w in live]
            continuity=[w.continuous for w in live]
            offline.append((record['label'],offline_flags,None))
            runtime.append((record['label'],live_flags,continuity))
            if len(live)!=len(record['windows']) or sustained(offline_flags)!=sustained(live_flags,continuity):
                divergent.append({'clip':record['source'],'label':record['label'],
                                  'offline_windows':len(record['windows']),'runtime_windows':len(live),
                                  'offline_sustained':sustained(offline_flags),
                                  'runtime_sustained':sustained(live_flags,continuity)})
        report['splits'][split]={
            'clips':len(rows),
            'clips_where_the_two_paths_differ':len(divergent),
            'offline_windows_the_card_measured':summarise(offline),
            'runtime_windows_the_app_emits':summarise(runtime),
            'divergent_clips':divergent}
    report['limits']=(
        'Window and clip level inside single-person clips, at the card threshold. Says nothing '
        'about who acted or when, and is not an alert rate per camera-hour. The runtime column '
        'is the emitted window set, not a full replay of the incident lifecycle.')
    destination=Path(arguments.out) if arguments.out else ROOT/'runs'/'window-parity.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2))
    print(json.dumps({split:{k:v for k,v in value.items() if k!='divergent_clips'}
                      for split,value in report['splits'].items()},indent=2))
    print(f'written: {destination.relative_to(ROOT)}')


if __name__=='__main__':main()
