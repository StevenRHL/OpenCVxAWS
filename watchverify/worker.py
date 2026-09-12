"""A single run owns all perception, temporal and incident state."""
from pathlib import Path
import argparse,csv,json,os,sqlite3,time,traceback
from datetime import datetime,timezone
import cv2
from . import jobs
from .perception import PoseEstimator,VideoExport,frames,video_info,sha256
from .core import Tracker,FeatureBuffer,RuleDetector,IncidentManager,FEATURE_NAMES
from .features import WindowAggregator,ACTIVITY_SUSTAINED_WINDOWS,ACTIVITY_CLEAR_WINDOWS
from .models import Models
ROOT=Path(__file__).resolve().parents[1]
BONES=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]

def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def run(run_id):
    job=jobs.get_job(run_id);folder=ROOT/'outputs'/run_id
    config=job.get('config',{});source=Path(job['source_path'])
    started=time.perf_counter();last_t=0.;encoder=None;db=None;all_events={};latest=[];models=Models()
    # Time the run analysed but could not observe a usable person. Reported as source
    # intervals so a reader can tell silence from absence of evidence.
    unobserved=[];unobserved_start=None
    metrics={'analysed_frames':0,'decoded_frames':0,'frames_with_pose':0,'valid_person_observations':0,'invalid_person_observations':0,'saturated_frames':0,'frames_without_usable_person':0,'track_gap_retirements':0,'ambiguous_track_retirements':0,'source_duration_s':None,'pose_seconds':0.,'feature_names':list(FEATURE_NAMES),'model_status':models.status,'warnings':[]}
    state='completed';reason='source_ended';error=None
    try:
        info=video_info(source);metrics['source_duration_s']=float(info['duration_s'])
        if info['duration_s']>600:raise ValueError('Please use a video no longer than 10 minutes.')
        if max(info['width'],info['height'])>1920 or min(info['width'],info['height'])>1080:raise ValueError('Please use video at 1920×1080 or smaller.')
        jobs.update_job(run_id,status='running',progress=0.,started_at_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),media=info,model_status=models.status,model_disclosures=models.disclosures())
        fps=float(config.get('analysis_fps',10));max_people=int(config.get('max_people',4));width=int(config.get('analysis_width',640))
        if not 1<=fps<=30 or not 1<=max_people<=8:raise ValueError('Invalid analysis configuration.')
        tracker=Tracker();buffers=FeatureBuffer()
        # Event timing comes from the trained fall card when one is present: the dwell
        # constants were selected on validation together with the posture threshold, and
        # using the code defaults with a trained model would apply an untuned pairing.
        timing={}
        if 'fall' in models.loaded:
            fall_card=models.loaded['fall'][1]
            for key in ('fall_hold','down_hold','recovery_hold'):
                if f'{key}_s' in fall_card:timing[key]=float(fall_card[f'{key}_s'])
        rules=RuleDetector(**timing);metrics['rule_timing']=timing
        activity_windows=WindowAggregator(*( [float(models.loaded['activity'][1].get('window_s',5.)),
                                              float(models.loaded['activity'][1].get('stride_s',1.))]
                                             if 'activity' in models.loaded else []))
        manager=IncidentManager(run_id,config.get('recording_start'))
        db=sqlite3.connect(folder/'events.db');db.execute('CREATE TABLE IF NOT EXISTS revisions(event_id TEXT, revision INTEGER, payload TEXT, PRIMARY KEY(event_id,revision))')
        encoder=VideoExport(folder/'annotated.partial.mp4',info['width'],info['height'],info['fps'])
        def persist(revisions):
            for event in revisions:
                payload=json.dumps(event,allow_nan=False);db.execute('INSERT OR IGNORE INTO revisions VALUES(?,?,?)',(event['event_id'],event['revision'],payload));all_events[event['event_id']]=event
            db.commit()
        # Runs of consecutive windows, not elapsed seconds: a window that never closed is
        # absent evidence, and time alone cannot tell it apart from evidence against.
        next_t=0.;last_analysis=-999.;last_progress=-1.;activity_run={};activity_normal_run={}
        sustained_windows=ACTIVITY_SUSTAINED_WINDOWS;clear_windows=ACTIVITY_CLEAR_WINDOWS
        if 'activity' in models.loaded:
            sustained_windows=int(models.loaded['activity'][1].get('sustained_windows',sustained_windows))
        metrics['activity_rule']={'sustained_windows':sustained_windows,'clear_windows':clear_windows}
        with PoseEstimator(config.get('pose_variant','full'),max_people,width) as estimator, (folder/'predictions.jsonl').open('w') as predictions:
            for index,t,bgr in frames(source):
                metrics['decoded_frames']+=1
                if (folder/'cancel.request').exists():state='cancelled';reason='cancelled';break
                if time.perf_counter()-started>3600:raise TimeoutError('The one-hour analysis limit was reached; partial evidence was saved.')
                if t+1e-6>=next_t:
                    tic=time.perf_counter();poses=estimator.detect(bgr,t);metrics['pose_seconds']+=time.perf_counter()-tic
                    tracks=tracker.update(poses,t)
                    for retired in tracker.retired_ids:buffers.reset(retired);rules.reset(retired);activity_windows.reset(retired);activity_run.pop(retired,None);activity_normal_run.pop(retired,None)
                    metrics['track_gap_retirements']+=len(tracker.gap_retired_ids);metrics['ambiguous_track_retirements']+=len(tracker.ambiguous_ids)
                    metrics['analysed_frames']+=1;metrics['frames_with_pose']+=bool(poses);metrics['saturated_frames']+=len(poses)>=max_people
                    candidates=[];valid=[];latest=tracks
                    for track_id,pose in tracks:
                        feat=buffers.update(track_id,pose,t)
                        if feat is None:metrics['invalid_person_observations']+=1
                        else:metrics['valid_person_observations']+=1;valid.append(track_id)
                        fall=activity=None
                        decision=feat;activity_events=[]
                        if feat is not None:
                            fall=models.score('fall',feat['vector'],feat['feature_mask'])
                            # The activity model reads a window of recent movement, not one
                            # frame. Before a full ungapped window exists there is no score
                            # and the branch abstains rather than guessing.
                            for window in activity_windows.update(track_id,t,feat['vector'],feat['feature_mask'],
                                                                 feat['hip_speed'],feat['angular_speed'],
                                                                 feat['down'],feat['upright'],feat['quality']):
                                activity=models.score('activity',window.descriptor)
                                if activity is None:continue
                                # A window that does not continue the previous one begins a
                                # new run. Scores either side of a break in observation are
                                # not consecutive evidence, however close together they fall.
                                if not window.continuous:activity_run.pop(track_id,None);activity_normal_run.pop(track_id,None)
                                if activity['positive']:
                                    activity_normal_run.pop(track_id,None)
                                    activity_run[track_id]=activity_run.get(track_id,0)+1
                                    if activity_run[track_id]>=sustained_windows:
                                        activity_events.append({'category':'unusual_activity','track_id':track_id,'score':activity['score'],'score_type':activity.get('score_type','uncalibrated_anomaly_score'),'observations':['unusual_motion_against_training_baseline','human_review_required']})
                                else:
                                    activity_run.pop(track_id,None)
                                    # Sustained ordinary movement closes an activity incident.
                                    # The manager ignores this when none is active.
                                    activity_normal_run[track_id]=activity_normal_run.get(track_id,0)+1
                                    if activity_normal_run[track_id]>=clear_windows:
                                        activity_events.append({'category':'activity_clear','track_id':track_id,'score':activity['score'],'score_type':activity.get('score_type','uncalibrated_anomaly_score'),'observations':['sustained_normal_activity']})
                                        activity_normal_run.pop(track_id,None)
                            if fall and fall['positive']:
                                decision=dict(feat,down=True,upright=False)
                        obs=rules.update(track_id,decision,t)
                        for c in obs:
                            c['track_id']=track_id
                            if fall and fall['positive'] and c['category'] in ('possible_fall','person_down'):
                                c['observations']=[v.replace('horizontal_posture','down_posture') for v in c['observations']]
                                c['observations'].append('trained_down_posture_support')
                        candidates.extend(obs)
                        # Only a closed window carries evidence. Frames between windows say
                        # nothing either way, which is why the run is counted in windows and
                        # advanced where they are scored rather than on every analysed frame.
                        candidates.extend(activity_events)
                        row={'t':t,'track_id':track_id,'valid':feat is not None,'fall_model':fall,'activity_model':activity}
                        if feat is not None:row.update({'features':feat['vector'].tolist(),'feature_mask':feat['feature_mask'].tolist(),'quality':float(feat['quality']),'angle':float(feat['angle']),'down':bool(feat['down'])})
                        predictions.write(json.dumps(row,allow_nan=False)+'\n')
                    if valid:
                        if unobserved_start is not None:unobserved.append([round(unobserved_start,3),round(t,3)]);unobserved_start=None
                    else:
                        metrics['frames_without_usable_person']+=1
                        if unobserved_start is None:unobserved_start=t
                    persist(manager.step(candidates,t,valid));last_analysis=t;next_t=t+1/fps
                # Do not carry stale skeletons across gaps. Decision-time overlay uses emitted state only.
                overlay=bgr.copy()
                if t-last_analysis<=.25:
                    for track_id,pose in latest:
                        for a,b in BONES:
                            if min(pose[a,3],pose[b,3])>=.5:cv2.line(overlay,tuple(pose[a,:2].astype(int)),tuple(pose[b,:2].astype(int)),(102,224,192),2)
                        visible=pose[pose[:,3]>=.5]
                        if len(visible):cv2.putText(overlay,f'Person {track_id}',tuple(visible[0,:2].astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.5,(102,224,192),1,cv2.LINE_AA)
                labels=[e['category'].replace('_',' ') for e in all_events.values() if e['status']=='active']
                quality=' | '.join(dict.fromkeys(labels)) if labels else ('Observing - review candidates only' if latest else 'No usable pose - visibility unknown')
                cv2.rectangle(overlay,(0,0),(overlay.shape[1],52),(21,27,36),-1)
                cv2.putText(overlay,f'{t:07.2f}s  {quality[:90]}',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1,cv2.LINE_AA)
                cv2.putText(overlay,'Experimental / decision-time replay / silent export',(10,43),cv2.FONT_HERSHEY_SIMPLEX,.38,(172,185,197),1,cv2.LINE_AA)
                encoder.write(overlay,t)
                # A decoded frame rejected by cancellation or a processing failure is
                # not part of the successfully processed source interval.
                last_t=t
                progress=min(.99,t/max(info['duration_s'],t+.1))
                if progress-last_progress>.025 or index%100==0:
                    jobs.update_job(run_id,progress=progress,summary={'analysed_frames':metrics['analysed_frames'],'events':len(all_events),'source_time_s':t});last_progress=progress
                    if (folder/'annotated.partial.mp4').exists() and (folder/'annotated.partial.mp4').stat().st_size>2*1024**3:raise ValueError('Output reached the 2 GiB per-run limit.')
            persist(manager.close_all(last_t,reason))
    except Exception as exc:
        state='failed';error=str(exc)
        (folder/'error.log').write_text(traceback.format_exc())
        if 'manager' in locals() and db:
            try:persist(manager.close_all(last_t,'processing_failed'))
            except Exception:pass
    finally:
        if encoder:
            try:encoder.close();(folder/'annotated.partial.mp4').replace(folder/'annotated.mp4')
            except Exception as exc:
                if state=='completed':state='failed';error=f'Video export could not finish: {exc}'
        if db:db.close()
        metrics['elapsed_s']=time.perf_counter()-started;metrics['analysis_fps']=metrics['analysed_frames']/max(metrics['elapsed_s'],.001)
        if unobserved_start is not None:unobserved.append([round(unobserved_start,3),round(last_t,3)])
        metrics['unobserved_intervals']=unobserved
        metrics['unobserved_source_s']=round(sum(b-a for a,b in unobserved),3)
        metrics['source_duration_processed_s']=last_t
        metrics['unprocessed_source_s']=round(max((metrics['source_duration_s'] or 0.)-last_t,0.),3)
        metrics['status']=state;metrics['error']=error
        metrics['pose_frame_coverage']=metrics['frames_with_pose']/max(metrics['analysed_frames'],1)
        metrics['source_sha256']=sha256(source)
        if metrics['source_duration_s'] and last_t<metrics['source_duration_s']-1.:
            metrics['warnings'].append(f"Only {last_t:.1f}s of a {metrics['source_duration_s']:.1f}s source was decoded; the remainder was never observed.")
        if unobserved:metrics['warnings'].append(f'No usable person was observed for {metrics["unobserved_source_s"]:.1f}s across {len(unobserved)} interval(s); those periods are unknown, not clear.')
        if metrics['ambiguous_track_retirements']:metrics['warnings'].append('Person identities were dropped at ambiguous crossings; evidence either side of those moments may belong to different people.')
        if metrics['saturated_frames']:metrics['warnings'].append('Pose capacity reached in some frames; additional people may not have been observed.')
        metrics['warnings'].append('Pose availability is not model accuracy. Absence of an alert does not establish safety.')
        events=list(all_events.values());atomic_json(folder/'events.json',events);atomic_json(folder/'metrics.json',metrics)
        fields=['event_id','category','track_id','source_start_s','source_end_s','emitted_source_s','occurred_at_utc','created_at_utc','status','reason','score','observations']
        with (folder/'events.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            writer.writerows([dict(e,observations='; '.join(str(v) for v in e.get('observations',[]))) for e in events])
        jobs.update_job(run_id,status=state,progress=1. if state=='completed' else job.get('progress',0),error=error,completed_at_utc=datetime.now(timezone.utc).isoformat(),summary={'events':len(events),'analysed_frames':metrics['analysed_frames'],'source_time_s':last_t,'elapsed_s':metrics['elapsed_s'],'pose_coverage':metrics['pose_frame_coverage'],'source_duration_s':metrics['source_duration_s'],'unobserved_source_s':metrics['unobserved_source_s'],'unobserved_intervals':len(unobserved),'unprocessed_source_s':metrics['unprocessed_source_s']})
    return state

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args();run(args.run_id)
