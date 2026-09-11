"""Score a frozen operating point against a held-out split, once.

Separated from training on purpose: training never reads the test split, and this script
never selects anything. Pass the operating point explicitly, or let it read the one
recorded in the model card.

Recall counts every in-scope event, including events missed because pose was never
available. Reporting recall only over successfully extracted poses would overstate it.
"""
from pathlib import Path
import argparse,datetime,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from watchverify.core import RuleDetector
from watchverify.models import Models
from watchverify.evaluation import (load_posture_labels,frame_times,fall_ground_truth,replay,
                                    match_fall_events,aggregate,wilson_interval,failure_gallery)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_models import fall_sequences,urfall_split,xy

def unobserved_seconds(attempts,reasons,duration_s):
    """Source seconds the extractor attempted but produced no usable observation for.

    Each attempt covers the span until the next attempt; the last one covers the remaining
    source time. Counting attempts instead of seconds would misreport corpora sampled at
    different intervals.
    """
    times=[float(a[0]) for a in attempts]
    total=0.
    for i,(t,reason) in enumerate(zip(times,reasons)):
        if reason=='ok':continue
        total+=(times[i+1]-t) if i+1<len(times) else max(duration_s-t,0.)
    return total


def fall_report(split,posture_threshold,down_hold,fall_hold,use_model=True,on_time_s=3.0,source='urfall'):
    labels=load_posture_labels(source=source)
    models=Models()
    score=None
    if use_model:
        if 'fall' not in models.loaded:raise SystemExit('No trained fall model; run train_models.py first')
        model,card=models.loaded['fall']
        def score(vector,mask):
            return float(model.predict_proba(np.concatenate([vector,mask.astype(float)]).reshape(1,-1))[0,1])
    per_sequence=[];results=[]
    for record in fall_sequences((source,)):
        if record['split']!=split:continue
        sid=record['sequence_id'];path=record['path']
        times=frame_times(sid,source=source)
        d=np.load(path,allow_pickle=False)
        reasons=[str(v) for v in d['attempt_reason']]
        coverage=reasons.count('ok')/max(len(reasons),1)
        unobserved=unobserved_seconds(d['attempts'],reasons,max(times.values()))
        truth=fall_ground_truth(sid,labels,times)
        events=replay(path,detector=RuleDetector(down_hold=down_hold,fall_hold=fall_hold),
                      score=score,threshold=posture_threshold)
        matched=match_fall_events(events,truth,max(times.values()),on_time_s=on_time_s,
                                  unobserved_s=unobserved)
        results.append(matched)
        per_sequence.append({'sequence':sid,'pose_coverage':round(coverage,4),
                             'unobserved_s':round(unobserved,3),
                             'true_events':len(truth),
                             'alerts':[{'category':c['category'],'t':round(c['t'],2)}
                                       for c in events if c['category']!='recovery'],
                             **{k:matched[k] for k in ('matched','on_time','missed','false_alerts','latencies_s')}})
    if not results:raise SystemExit(f'No sequences in split {split!r}')
    summary=aggregate(results)
    summary['recall_95ci']=wilson_interval(summary['matched'],summary['true_events'])
    summary['on_time_recall_95ci']=wilson_interval(summary['on_time'],summary['true_events'])
    summary['events_missed_with_low_pose_coverage']=sum(
        1 for r in per_sequence if r['missed'] and r['pose_coverage']<.9)
    return {'split':split,'source':source,'posture_threshold':posture_threshold,
            'down_hold_s':down_hold,'fall_hold_s':fall_hold,
            'model_used':bool(use_model),'on_time_horizon_s':on_time_s,
            'summary':summary,'failure_gallery':failure_gallery(per_sequence),
            'per_sequence':per_sequence}


def activity_report(split):
    """Held-out result for the activity branch, at the card's frozen threshold.

    Eligibility and split assignment come from `train_models.mnnit_eligible` rather than a
    second copy of the filter, so a clip cannot be eligible here and excluded there.

    Both clip rules are reported. `any_window` is the loose reading; `sustained` is what
    the worker actually emits, which needs consecutive flagged windows.
    """
    from train_models import mnnit_eligible,ACTIVITY_SUSTAINED_WINDOWS
    card=json.loads((ROOT/'models/activity.json').read_text())
    models=Models()
    if 'activity' not in models.loaded:raise SystemExit('No trained activity model')
    eligible=mnnit_eligible()[0]
    rows=[r for r in eligible if r['split']==split]
    tp=fp=fn=tn=0;clip_flags=[]
    for r in rows:
        flags=[bool(models.score('activity',v)['positive']) for v in r['windows']]
        for f in flags:
            if r['label']==1:tp+=f;fn+=not f
            else:fp+=f;tn+=not f
        run=best=0
        for f in flags:
            run=run+1 if f else 0;best=max(best,run)
        clip_flags.append({'clip':r['source'],'label':r['label'],'windows':len(flags),
                           'flagged':int(sum(flags)),'longest_flagged_run':best,
                           'sustained':best>=ACTIVITY_SUSTAINED_WINDOWS})
    def rate(label,key):
        members=[c for c in clip_flags if c['label']==label]
        return round(sum(1 for c in members if c[key])/max(len(members),1),4)
    return {'branch':'activity','split':split,'threshold':card['threshold'],
            'window_s':card.get('window_s'),'clips':len(rows),
            'windows':tp+fp+fn+tn,'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'window_precision':round(tp/max(tp+fp,1),4),'window_recall':round(tp/max(tp+fn,1),4),
            'flagged_normal_window_fraction':round(fp/max(fp+tn,1),4),
            'clip_rule_any_window':{'shoplifting_clips_alerted':rate(1,'flagged'),
                                    'normal_clips_alerted':rate(0,'flagged')},
            'clip_rule_sustained':{'consecutive_windows_required':ACTIVITY_SUSTAINED_WINDOWS,
                                   'shoplifting_clips_alerted':rate(1,'sustained'),
                                   'normal_clips_alerted':rate(0,'sustained')},
            'recall_95ci':wilson_interval(tp,tp+fn),
            'limits':'Window level inside single-person clips. Says nothing about who acted or when.'}


def cross_domain_fall_report(split,posture_threshold,down_hold,fall_hold):
    """Fall false alerts on retail footage — the known cross-domain failure.

    Every MNNIT clip is a fall negative under the same weak assumption used in training:
    these clips are believed to contain nobody lying on the floor. Shoplifting clips are
    reported separately because none of them are ever used as training negatives, so they
    are the cleaner evidence.
    """
    from train_models import mnnit_eligible
    models=Models()
    if 'fall' not in models.loaded:raise SystemExit('No trained fall model')
    model,_=models.loaded['fall']
    def score(vector,mask):
        return float(model.predict_proba(np.concatenate([vector,mask.astype(float)]).reshape(1,-1))[0,1])
    out={}
    for scene in ('normal','shoplifting'):
        label=1 if scene=='shoplifting' else 0
        clips=[r for r in mnnit_eligible()[0] if r['label']==label and r['split']==split]
        alerted=0;seconds=0.
        for r in clips:
            path=ROOT/'data/processed/features/mnnit'/f"{r['source']}.npz"
            with np.load(path,allow_pickle=False) as d:
                seconds+=float(d['t'][-1]-d['t'][0]) if len(d['t']) else 0.
            events=replay(path,detector=RuleDetector(down_hold=down_hold,fall_hold=fall_hold),
                          score=score,threshold=posture_threshold)
            alerted+=any(c['category'] in ('possible_fall','person_down') for c in events)
        out[scene]={'clips':len(clips),'clips_with_a_fall_alert':alerted,
                    'rate':round(alerted/max(len(clips),1),4),'source_seconds':round(seconds,1),
                    'used_as_fall_training_negatives':scene=='normal' and split=='train'}
    out['note']=('No annotation confirms that these clips contain nobody on the floor; a genuine '
                 'person-down in a retail clip would be counted here as a false alert.')
    return out


def _shown(path):
    """Repo-relative when it is inside the repo, absolute otherwise — `--out` may point
    anywhere, and printing the path must not be able to fail."""
    try:return str(path.relative_to(ROOT))
    except ValueError:return str(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split',default='test',choices=['train','validation','test'])
    p.add_argument('--posture-threshold',type=float)
    p.add_argument('--down-hold',type=float)
    p.add_argument('--fall-hold',type=float)
    p.add_argument('--branch',default='both',choices=['fall','activity','both'])
    p.add_argument('--source',default='urfall',help='Fall corpus to score: urfall or owncam')
    p.add_argument('--on-time-s',type=float,default=3.0)
    p.add_argument('--rules-only',action='store_true',help='measure the hand-written rules instead')
    p.add_argument('--out')
    a=p.parse_args()
    card_path=ROOT/'models/fall.json'
    threshold=a.posture_threshold;hold=a.down_hold;fall_hold=a.fall_hold
    if not a.rules_only and card_path.exists():
        card=json.loads(card_path.read_text())
        threshold=threshold if threshold is not None else card['threshold']
        hold=hold if hold is not None else card.get('down_hold_s',RuleDetector().down_hold)
        fall_hold=fall_hold if fall_hold is not None else card.get('fall_hold_s',RuleDetector().fall_hold)
    threshold=threshold if threshold is not None else .5
    hold=hold if hold is not None else RuleDetector().down_hold
    fall_hold=fall_hold if fall_hold is not None else RuleDetector().fall_hold
    report=fall_report(a.split,threshold,hold,fall_hold,use_model=not a.rules_only,
                       on_time_s=a.on_time_s,source=a.source)
    report['generated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    destination=Path(a.out) if a.out else ROOT/'runs'/f'evaluation-{a.source}-{a.split}.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2))
    print(json.dumps({'branch':'fall','split':a.split,'source':a.source,'rules_only':a.rules_only,
                      'posture_threshold':threshold,'fall_hold_s':fall_hold,'down_hold_s':hold,
                      **report['summary'],'report':_shown(destination)},indent=2))
    if a.branch in ('activity','both'):
        print(json.dumps(activity_report(a.split),indent=2))
    if a.branch in ('fall','both'):
        print(json.dumps({'branch':'fall_cross_domain','split':a.split,
                          **cross_domain_fall_report(a.split,threshold,hold,fall_hold)},indent=2))

if __name__=='__main__':main()
