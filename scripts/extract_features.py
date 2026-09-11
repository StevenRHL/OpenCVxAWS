"""Cache causal runtime features from real videos, preserving missing-observation frames.

A cache is keyed by everything that can change its contents — source video, pose asset,
extractor settings, feature schema and library versions — not by the video alone. A
changed setting forces recomputation rather than silently reusing stale features.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from watchverify.perception import PoseEstimator,frames,sha256
from watchverify.core import Tracker,FeatureBuffer,FEATURE_NAMES,FEATURE_SCHEMA_VERSION
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'data/manifests/features.jsonl'
EXTRACTOR_VERSION=2
SAMPLE_INTERVAL_S=.1
POSE_VARIANT='full'
ANALYSIS_WIDTH=640
REASONS=('ok','no_pose','pose_geometry_rejected','features_rejected')

def library_versions():
    import av,cv2,mediapipe
    return {'mediapipe':mediapipe.__version__,'av':av.__version__,'cv2':cv2.__version__,'numpy':np.__version__}

def sidecar_mapping(path):
    """decoded_index -> original source frame id, from the preparation sidecar."""
    sidecar=path.with_suffix('.json')
    if not sidecar.exists():return None,None
    card=json.loads(sidecar.read_text())
    rows=card.get('frames') or []
    if not rows or 'decoded_index' not in rows[0] or 'source_frame' not in rows[0]:
        raise ValueError(f'{path.name}: sidecar has no decoded_index/source_frame mapping; re-run prepare')
    return {int(r['decoded_index']):int(r['source_frame']) for r in rows},card

def build_config(path,card,estimator,tracker,buffer,max_people,interval,width,variant):
    config={'extractor_version':EXTRACTOR_VERSION,
            'source':str(path.relative_to(ROOT)),'source_sha256':sha256(path),
            'archive_sha256':(card or {}).get('archive_sha256'),
            'preparer_version':(card or {}).get('preparer_version'),
            'pose_variant':variant,'pose_asset_sha256':estimator.asset_sha256,
            'analysis_width':width,'max_people':max_people,'sample_interval_s':interval,
            'pose_confidence':{'detection':estimator.detection_confidence,
                               'presence':estimator.presence_confidence,
                               'tracking':estimator.tracking_confidence},
            'tracker':{k:getattr(tracker,k) for k in ('max_gap','max_distance','ambiguity_margin','confidence')},
            'feature_buffer':{k:getattr(buffer,k) for k in ('window_s','max_gap','confidence')},
            'feature_schema_version':FEATURE_SCHEMA_VERSION,'feature_names':list(FEATURE_NAMES),
            'libraries':library_versions()}
    blob=json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False)
    return config,hashlib.sha256(blob.encode()).hexdigest()

def extract(path,source='urfall',max_people=1,interval=SAMPLE_INTERVAL_S,width=ANALYSIS_WIDTH,variant=POSE_VARIANT,force=False):
    cache=ROOT/'data/processed/features'/source/f'{path.stem}.npz';cache.parent.mkdir(parents=True,exist_ok=True)
    mapping,card=sidecar_mapping(path)
    frame_id_source='sidecar_mapping' if mapping else 'decoded_index'
    tic=time.perf_counter();tracker=Tracker();buffer=FeatureBuffer()
    with PoseEstimator(variant,max_people,width) as estimator:
        config,config_key=build_config(path,card,estimator,tracker,buffer,max_people,interval,width,variant)
        if cache.exists() and not force:
            with np.load(cache,allow_pickle=False) as old:
                if 'config_sha256' in old.files and str(old['config_sha256'])==config_key:
                    return cache,{'source':config['source'],'status':'cached'}
        records=[];attempts=[];reasons=[];attempt_frames=[];next_t=0
        for index,t,bgr in frames(path):
            if t+1e-6<next_t:continue
            if mapping is not None and index not in mapping:
                raise ValueError(f'{path.name}: decoded frame {index} is not in the preparation mapping')
            source_frame=mapping[index] if mapping is not None else index
            poses=estimator.detect(bgr,t);tracks=tracker.update(poses,t);valid_count=0
            for retired in tracker.retired_ids:buffer.reset(retired)
            for track,pose in tracks:
                feature=buffer.update(track,pose,t)
                if feature is None:continue
                valid_count+=1
                records.append((t,index,source_frame,track,feature['vector'],feature['feature_mask'],feature['down'],feature['upright'],feature['hip_speed'],feature['angular_speed'],feature['quality']))
            if valid_count:reason='ok'
            elif not poses:reason='no_pose'
            elif not tracks:reason='pose_geometry_rejected'
            else:reason='features_rejected'
            attempts.append((t,index,len(poses),valid_count));reasons.append(reason);attempt_frames.append(source_frame)
            next_t=t+interval
    n=len(records);elapsed=time.perf_counter()-tic
    np.savez_compressed(cache,
        config_sha256=config_key,config_json=json.dumps(config,sort_keys=True),
        source_sha256=config['source_sha256'],source=config['source'],
        feature_schema_version=FEATURE_SCHEMA_VERSION,feature_names=np.array(FEATURE_NAMES),
        frame_id_source=frame_id_source,
        t=np.array([r[0] for r in records]),
        frame=np.array([r[1] for r in records],dtype=int),
        source_frame=np.array([r[2] for r in records],dtype=int),
        track=np.array([r[3] for r in records],dtype=int),
        x=np.array([r[4] for r in records]).reshape(n,12),
        mask=np.array([r[5] for r in records],dtype=bool).reshape(n,12),
        down=np.array([r[6] for r in records],dtype=bool),
        upright=np.array([r[7] for r in records],dtype=bool),
        hip_speed=np.array([r[8] for r in records]),
        angular_speed=np.array([r[9] for r in records]),
        quality=np.array([r[10] for r in records]),
        attempts=np.array(attempts).reshape(-1,4),
        attempt_source_frame=np.array(attempt_frames,dtype=int),
        attempt_reason=np.array(reasons),
        elapsed_s=elapsed)
    counts={r:int(sum(v==r for v in reasons)) for r in REASONS}
    summary={'source':config['source'],'status':'extracted','config_sha256':config_key[:12],
             'frame_id_source':frame_id_source,'rows':n,'sampled_frames':len(attempts),
             'usable_frames':counts['ok'],'missing_observation_reasons':counts,
             'seconds':round(elapsed,2)}
    MANIFEST.parent.mkdir(parents=True,exist_ok=True)
    with MANIFEST.open('a') as f:
        f.write(json.dumps({'cache':str(cache.relative_to(ROOT)),'written_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                            'config_sha256':config_key,'config':config,**{k:summary[k] for k in ('rows','sampled_frames','usable_frames','missing_observation_reasons','frame_id_source')}})+'\n')
    return cache,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='urfall');p.add_argument('--max-people',type=int,default=1)
    p.add_argument('--interval',type=float,default=SAMPLE_INTERVAL_S)
    p.add_argument('--width',type=int,default=ANALYSIS_WIDTH)
    p.add_argument('--variant',default=POSE_VARIANT);p.add_argument('--force',action='store_true')
    a=p.parse_args()
    folder=ROOT/'data/processed'/a.source
    # Leading-underscore directories hold partials and provenance trees, not prepared data.
    paths=sorted(v for v in folder.rglob('*.mp4') if not any(part.startswith('_') for part in v.relative_to(folder).parts))
    if not paths:raise SystemExit(f'No prepared videos in {folder}')
    totals={'extracted':0,'cached':0}
    for path in paths:
        _,summary=extract(path,a.source,a.max_people,a.interval,a.width,a.variant,a.force)
        totals[summary['status']]+=1;print(json.dumps(summary),flush=True)
    print(json.dumps({'videos':len(paths),**totals},indent=2))

if __name__=='__main__':main()
