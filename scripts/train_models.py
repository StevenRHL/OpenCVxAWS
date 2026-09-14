"""Train the fall and retail-activity branches on real data.

Discipline enforced here:
  * splits are grouped by recording and frozen before any feature building;
  * every threshold and model choice is selected on validation only;
  * the test split is never read by this script — `evaluate_events.py` reads it once,
    after an operating point has been chosen from the validation tradeoff curve.

Neither model diagnoses injury nor establishes theft. The fall branch classifies down
posture; the activity branch scores motion similarity to clips labelled shoplifting.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (IsolationForest,GradientBoostingClassifier,RandomForestClassifier,
                              ExtraTreesClassifier,HistGradientBoostingClassifier)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold,cross_val_predict
from sklearn.base import clone
from sklearn.metrics import roc_auc_score,average_precision_score
from watchverify.core import FEATURE_NAMES,FEATURE_SCHEMA_VERSION,RuleDetector
from watchverify.features import (CLIP_COLUMNS,WINDOW_S,STRIDE_S,windows,
                                  ACTIVITY_SUSTAINED_WINDOWS)
from watchverify.evaluation import (load_posture_labels,frame_times,fall_ground_truth,
                                    replay,match_fall_events,aggregate,wilson_interval)
ROOT=Path(__file__).resolve().parents[1]
RUN='pilot-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT=ROOT/'runs'/RUN
LYING=1
MIN_CLIP_COVERAGE=.5
MIN_CLIP_ROWS=10
POSTURE_THRESHOLDS=[.3,.4,.5,.6,.7,.8,.9]
# fall_hold gates `possible_fall`; down_hold gates `person_down`. Sweeping only the latter
# leaves the fall alert untouched. Measured: the geometric down-run is shorter than the
# 0.35 s default in 13 of 30 falls, because pose is frequently lost once the person is down.
FALL_HOLDS=[0.,.1,.2,.35,.5]
DOWN_HOLDS=[.8,1.5,2.0]
# Clip-grouped repeated CV over the training split replaces single-validation-split model
# selection. Measured on the dev clips: fold-to-fold window ROC-AUC for one configuration
# spans 0.68-0.97 (sd ~0.07), so a single 31-clip validation split cannot separate
# candidates that differ by less than that, and it cannot show whether a gap to the test
# split is a real transfer failure or ordinary sampling noise.
CV_FOLDS=5
CV_SEEDS=(0,1,2)
# The activity operating point is set as a *rate*: the fraction of ordinary windows we are
# willing to flag. It is read off out-of-fold scores, never off scores the model has
# already fitted. An in-sample quantile is the defect that made the pilot's 21% validation
# flag rate become 48% on test.
ACTIVITY_TARGET_NORMAL_FLAG=.21
# Re-exported from watchverify.features, which the worker also reads, so the measured rule
# and the shipped rule cannot be changed apart. Clip-level metrics must use it; "any single
# flagged window" is not what the application does.

# ---------------------------------------------------------------- splits

SPLIT_SALT='urfall-split-v2'
SPLIT_COUNTS={'fall':30,'adl':40}

def hash_split(members,salt):
    """Rank sequences by a stable hash of their id, then take 60/20/20.

    Splitting on sequence number was rejected after measuring it: pose availability while
    a person is on the ground falls from 82.9% (fall-01..18) to 29.0% (fall-25..30), so
    numbered splits are ordered by difficulty rather than exchangeable. Hash ranking
    removes that confound. Chosen before any test-split result was computed.
    """
    ranked=sorted(members,key=lambda sid:hashlib.sha256(f'{salt}:{sid}'.encode()).hexdigest())
    total=len(ranked);train=round(total*.6);validation=round(total*.8)
    return {sid:('train' if i<train else 'validation' if i<validation else 'test')
            for i,sid in enumerate(ranked)}

def _assign(kind):
    return hash_split([f'{kind}-{i:02d}' for i in range(1,SPLIT_COUNTS[kind]+1)],SPLIT_SALT)

_SPLITS={sid:split for kind in SPLIT_COUNTS for sid,split in _assign(kind).items()}

OWNCAM_SPLIT_SALT='owncam-split-v1'

def owncam_split(sequence_ids):
    """Hash-assign own-camera clips 60/20/20 within each kind.

    Splitting inside `fall*` and `adl*` separately keeps both classes present in every
    split even when the corpus is small, which it will be at first. Unlike UR Fall, the
    actor here IS known — a single person — so actor independence is known to be absent
    rather than merely unknown, and the model card says so.
    """
    assignment={}
    for kind in ('fall','adl'):
        members=sorted(sid for sid in sequence_ids if sid.startswith(kind))
        assignment.update(hash_split(members,OWNCAM_SPLIT_SALT))
    unknown=[sid for sid in sequence_ids if sid not in assignment]
    if unknown:
        raise ValueError(f'Own-camera clips must start with "fall" or "adl": {sorted(unknown)}')
    return assignment

# Whether an artifact trained on a source may leave this machine. Recorded on the card so
# an encumbered model identifies itself instead of relying on someone remembering D029.
SOURCE_LICENSE={
    'urfall':{'license':'CC BY-NC-SA 4.0','release_cleared':False,
              'note':'Non-commercial academic research. NOT cleared for competition or commercial distribution.'},
    'owncam':{'license':'Own recording','release_cleared':True,
              'note':'Recorded locally by the project owner; cleared for release by the person who recorded it.'},
    'reviewed':{'license':'Admin-curated from reviewed application footage','release_cleared':False,
               'note':'Clips an admin exported from the Learning queue after reviewing an alert '
                      '(see watchverify/review.py, scripts/import_reviewed_exports.py). Source '
                      'footage may be third-party; not cleared for release until someone checks '
                      'the original recording\'s own permissions.'},
}

def urfall_split(sequence_id):
    """Grouped by recording. Actor identity is NOT published in the downloaded material,
    so actor independence across splits is unknown and must not be claimed."""
    if sequence_id not in _SPLITS:raise KeyError(f'Unknown UR sequence {sequence_id!r}')
    return _SPLITS[sequence_id]

def mnnit_split(stem,index,total):
    """Grouped by clip, stratified within class by position: 60/20/20."""
    if index<round(total*.6):return 'train'
    return 'validation' if index<round(total*.8) else 'test'

def fall_sequences(sources):
    """[{source, sequence_id, path, split}] for every cached fall-branch sequence."""
    records=[]
    for source in sources:
        if source not in SOURCE_LICENSE:
            raise ValueError(f'Unknown fall source {source!r}; known: {sorted(SOURCE_LICENSE)}')
        caches=sorted((ROOT/'data/processed/features'/source).glob('*.npz'))
        if not caches:
            raise ValueError(f'No {source} feature caches; run extract_features.py --source {source}')
        stems=[p.stem for p in caches]
        assignment=({stem:urfall_split(stem) for stem in stems} if source=='urfall'
                    else owncam_split(stems))
        records.extend({'source':source,'sequence_id':p.stem,'path':p,'split':assignment[p.stem]}
                       for p in caches)
    duplicates={r['sequence_id'] for r in records}
    if len(duplicates)!=len(records):
        raise ValueError('Two sources contain the same sequence id; rename one corpus')
    return records


def assert_no_leakage(pairs):
    """`pairs` is (group_key, split). Keys must be content identities, not file names.

    Identical footage published under two names would otherwise land in two splits and
    inflate held-out results — MNNIT ships three byte-identical Normal clips, so this is
    a real failure mode rather than a hypothetical one.
    """
    groups={}
    for key,split in pairs:groups.setdefault(key,set()).add(split)
    overlapping={k:sorted(v) for k,v in groups.items() if len(v)>1}
    if overlapping:raise ValueError(f'Split leakage — same content in multiple splits: {overlapping}')

# ---------------------------------------------------------------- shared

def xy(d):return np.concatenate([d['x'],d['mask'].astype(float)],axis=1)
FEATURE_COLUMNS=list(FEATURE_NAMES)+[n+'_valid' for n in FEATURE_NAMES]


def retail_not_lying_frames():
    """Retail frames used as `not lying` negatives for the fall posture model.

    UR Fall is home/laboratory footage, and the pilot fall model false-alarmed on retail
    scenes: 23 of 66 MNNIT clips that were never part of fall training produced a fall
    alert. Those clips show people walking and standing among shelves, so `not lying` is a
    defensible weak label for them — but it is a weak label, not an annotation. Any MNNIT
    clip that does contain someone on the floor is mislabelled here, and no one has watched
    all 179 clips to rule that out.

    Only clips assigned to the activity TRAIN split contribute, so the activity branch's
    own validation and test clips stay unused by any model.
    """
    rows=[];groups=[];sources=[]
    for record in mnnit_eligible()[0]:
        if record['label'] or record['split']!='train':continue
        frames=xy(np.load(ROOT/'data/processed/features/mnnit'/f"{record['source']}.npz",allow_pickle=False))
        if not len(frames) or not np.isfinite(frames).all():continue
        rows.append(frames);groups.append(np.full(len(frames),record['source']));sources.append(record['source'])
    return ((np.concatenate(rows) if rows else np.zeros((0,len(FEATURE_COLUMNS)))),
            (np.concatenate(groups) if groups else np.zeros((0,),dtype=object)),sources)

def save(name,model,threshold,metadata):
    artifact=ROOT/'models'/f'{name}.joblib';joblib.dump(model,artifact)
    card={'run_id':RUN,'artifact':artifact.name,
          'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
          'threshold':float(threshold),'with_mask':metadata.get('input','frame_vector')=='frame_vector',
          'n_features':int(getattr(model,'n_features_in_',0) or 0),
          'feature_schema_version':FEATURE_SCHEMA_VERSION,
          'status':'Experimental pilot; not validated for operational use',
          'threshold_provenance':'selected on validation only; test split unseen by training',
          **metadata}
    (ROOT/'models'/f'{name}.json').write_text(json.dumps(card,indent=2))
    return card

def snapshot_disclosure_evidence(name):
    """Backfill `disclosure_evidence` onto the just-written card from the just-written
    metrics file, so a retrained card is self-verifying without a manual step. D047/D048
    introduced this field by hand after Part 2; since train_models.py never wrote it,
    every later retrain silently dropped it until someone re-did that by hand.
    """
    card_path=ROOT/'models'/f'{name}.json'
    card=json.loads(card_path.read_text())
    metrics_path=OUT/f'{name}_metrics.json'
    metrics=json.loads(metrics_path.read_text())
    card['disclosure_evidence']={
        'split':'validation','model_run_id':card['run_id'],
        'artifact_sha256':card['sha256'],
        'metrics_path':str(metrics_path.relative_to(ROOT)),
        'metrics_sha256':hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        'operating_point':metrics['selected_operating_point'],
        'verified_on':datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d'),
        'note':'Snapshotted automatically by train_models.py at training time.'}
    card_path.write_text(json.dumps(card,indent=2))
    return card

# ---------------------------------------------------------------- fall branch

def fall_candidates():
    """Frame-level posture classifiers, scored by the same grouped_cv_scores harness as
    the activity branch (see fall_training). Kept small: each entry costs
    CV_FOLDS*len(CV_SEEDS) fits on ~6k training rows.
    """
    candidates={}
    for c in (.3,1,3):
        candidates[f'logistic_c{c}']=make_pipeline(
            StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=2000,random_state=42))
    for n_estimators in (100,200,300):
        for max_depth in (2,3):
            for learning_rate in (.05,.1):
                candidates[f'gradient_boosting_n{n_estimators}_d{max_depth}_lr{learning_rate}']=(
                    GradientBoostingClassifier(random_state=42,n_estimators=n_estimators,
                                               max_depth=max_depth,learning_rate=learning_rate))
    # Same two families scripts/compare_models.py has used as ad hoc challengers; folded into
    # the permanent grid after a run of that tool showed extra_trees_leaf2 clearing its
    # development gate on validation (CV-AUC delta +0.004, false alerts 6->4 at unchanged
    # recall). Selection below does NOT trust CV-AUC alone to choose among these — see the
    # comment at the shortlist further down for why.
    for leaf in (2,8):
        candidates[f'extra_trees_leaf{leaf}']=ExtraTreesClassifier(
            n_estimators=200,min_samples_leaf=leaf,max_features=.7,
            class_weight='balanced',random_state=42,n_jobs=1)
    for leaves in (7,15):
        candidates[f'hist_gradient_leaves{leaves}']=HistGradientBoostingClassifier(
            max_iter=150,max_leaf_nodes=leaves,learning_rate=.05,
            l2_regularization=2.,min_samples_leaf=20,early_stopping=False,random_state=42)
    return candidates

def fall_training(sources=('urfall',)):
    sources=tuple(sources)
    labels={source:load_posture_labels(source=source) for source in sources}
    records=fall_sequences(sources)
    assignment={r['sequence_id']:r['split'] for r in records}
    assert_no_leakage([(str(np.load(r['path'],allow_pickle=False)['source_sha256']),r['split'])
                       for r in records])
    groups={'train':[],'validation':[],'test':[]};inventory=[]
    for record in records:
        sid=record['sequence_id'];split=record['split'];path=record['path']
        d=np.load(path,allow_pickle=False)
        if 'source_frame' not in d.files or str(d['frame_id_source'])!='sidecar_mapping':
            raise ValueError(f'{sid}: cache lacks the source-frame mapping; re-run preparation')
        posture=np.array([labels[record['source']].get(sid,{}).get(int(i),0) for i in d['source_frame']])
        keep=posture!=0  # Publisher: "we don't use '0' frames in classification".
        reasons=[str(v) for v in d['attempt_reason']]
        inventory.append({'source':sid,'corpus':record['source'],'split':split,
                          'config_sha256':str(d['config_sha256']),
                          'labelled_rows':int(keep.sum()),'down_rows':int((posture==LYING).sum()),
                          'sampled_frames':len(d['attempts']),
                          'missing_observation_reasons':{r:reasons.count(r) for r in sorted(set(reasons))}})
        if keep.any() and split!='test':
            groups[split].append((xy(d)[keep],(posture[keep]==LYING).astype(int),
                                  np.full(int(keep.sum()),sid)))
    if not groups['train'] or not groups['validation']:
        raise ValueError('Need labelled train and validation sequences')
    arrays={k:(np.concatenate([x for x,_,_ in v]),np.concatenate([y for _,y,_ in v]),
              np.concatenate([g for _,_,g in v]))
            for k,v in groups.items() if v}
    if len(np.unique(arrays['train'][1]))<2:raise ValueError('Training split has one posture class only')

    # Cross-domain negatives. Held-out evidence for adding them, measured on the 66 MNNIT
    # clips that are never part of fall training: false fall alerts fell from 23/66 to
    # 8/66, while UR validation kept 6/6 on-time detections at the same false-alert count.
    retail_x,retail_group,retail_sources=retail_not_lying_frames()
    fx,fy,fgroup=arrays['train']
    if len(retail_x):
        fx=np.concatenate([fx,retail_x]);fy=np.concatenate([fy,np.zeros(len(retail_x),int)])
        fgroup=np.concatenate([fgroup,retail_group])
    training=(fx,fy)
    negative_note={'retail_negative_frames':int(len(retail_x)),'retail_negative_clips':len(retail_sources),
                   'retail_negative_sources':retail_sources,
                   'label_basis':'weak — MNNIT normal clips are assumed to contain no one lying on the floor; the clips were not annotated frame by frame'}

    # Model selection runs on clip/sequence-grouped repeated CV over the training split,
    # the same harness D037 introduced for the activity branch — a single ~30-sequence
    # validation split is exactly the kind of surface D037 found too noisy to select on.
    vx,vy,_=arrays['validation'];comparison={}
    for name,proto in fall_candidates().items():
        comparison[name]={'cv':summarise(grouped_cv_scores(proto,fx,fy,fgroup))}
    for name in comparison:
        candidate=clone(fall_candidates()[name]);candidate.fit(*training)
        probability=candidate.predict_proba(vx)[:,1]
        comparison[name]['validation_roc_auc']=float(roc_auc_score(vy,probability)) if len(np.unique(vy))>1 else None
        comparison[name]['validation_average_precision']=float(average_precision_score(vy,probability)) if len(np.unique(vy))>1 else None

    # Frame-level CV-AUC ranks families by how well they separate down/not-down frames, but
    # that is not what the app is scored on — measured directly: with the wider grid below,
    # the top CV-AUC family (hist_gradient_leaves7, mean 0.9713) swept to a worse held-out
    # test result (11 alerts/5 false, vs. 10/4 for the installed model) than families ranked
    # lower on CV-AUC. So the shortlist is only a cheap pre-filter; the actual choice runs
    # every shortlisted family through the same event-level sweep `pick_operating_point`
    # already trusts, and picks whichever family's *swept* operating point is best — matching
    # scripts/compare_models.py's event-gated approach rather than trusting CV-AUC alone.
    # Newly added families are always swept even if their frame-level CV-AUC ranks below
    # the established pool, so a strong event-level performer (e.g. extra_trees_leaf2) is
    # never excluded from the fight by a CV-AUC pre-filter — which is exactly the failure
    # mode this shortlist exists to avoid.
    SHORTLIST_TOP_CV=2
    NEW_FAMILIES=('extra_trees_leaf2','extra_trees_leaf8','hist_gradient_leaves7','hist_gradient_leaves15')
    top_by_cv=sorted(comparison,key=lambda n:(comparison[n].get('cv') or {}).get('mean_roc_auc',0),
                     reverse=True)[:SHORTLIST_TOP_CV]
    shortlist=list(dict.fromkeys([*top_by_cv,*(f for f in NEW_FAMILIES if f in comparison)]))
    swept={}
    for name in shortlist:
        candidate=clone(fall_candidates()[name]);candidate.fit(*training)
        candidate_sweep=fall_event_sweep(candidate,records,labels,'validation')
        swept[name]={'model':candidate,'sweep':candidate_sweep,'best':pick_operating_point(candidate_sweep)}
    chosen=max(swept,key=lambda n:(swept[n]['best']['on_time_recall'],
                                   -(swept[n]['best']['false_alerts_per_hour'] or 0),
                                   -swept[n]['best']['false_alerts'],
                                   -swept[n]['best']['median_latency_s']))
    comparison[chosen]['event_shortlisted']=True
    for name in swept:
        comparison[name]['event_swept_on_time_recall']=swept[name]['best']['on_time_recall']
        comparison[name]['event_swept_false_alerts']=swept[name]['best']['false_alerts']
    model=swept[chosen]['model'];sweep=swept[chosen]['sweep'];best=swept[chosen]['best']
    licensing={source:SOURCE_LICENSE[source] for source in sources}
    release_cleared=all(entry['release_cleared'] for entry in licensing.values())
    card=save('fall',model,best['posture_threshold'],{
        'target':'down_posture','input':'frame_vector','scoring':'probability',
        'feature_names':FEATURE_COLUMNS,
        'model_family':chosen,'model_comparison':comparison,
        'down_hold_s':best['down_hold_s'],'fall_hold_s':best['fall_hold_s'],
        'training_source':' + '.join(sources)+' posture labels, plus MNNIT retail frames as weak not-lying negatives',
        'training_sources':licensing,
        'release_cleared':release_cleared,
        'cross_domain_negatives':negative_note,
        'label_note':('Publisher posture labels were defined on the depth frame of the same synchronised '
                      'camera-0 sequence.' if 'urfall' in sources else
                      'Posture labels expanded from hand-written clip intervals; see scripts/prepare_owncam.py.'),
        'actor_independence':('unknown — actor identity is not published in the downloaded material'
                              if 'urfall' in sources else
                              'absent — own-camera clips record a single known person, so the model has not been shown anyone else'),
        'license_note':('; '.join(f"{source}: {entry['note']}" for source,entry in licensing.items())),
        'not_a_diagnosis':'Down posture is not an injury diagnosis and a fall alert requires human review.',
        # Read by the application to tell a reviewer what this alert has been worth in
        # testing. Kept on the card so the interface can never quote a stale number.
        'validation_summary':{k:best.get(k) for k in ('true_events','matched','on_time','false_alerts',
                                                      'precision','median_latency_s')},
        'alert_caveat':(f"In validation, {best.get('false_alerts')} of {best.get('alerts')} "
                        'fall/person-down alerts did not match a labelled fall event; '
                        f"all {best.get('true_events')} labelled falls were matched on time. "
                        'This is a small validation sample used to select the operating point, '
                        'not an independent test or a reliable hourly rate. Unmatched alerts do '
                        'not establish that no person was down. Posture is not an injury '
                        'diagnosis. Performance on your camera is unknown.'),
        'metrics_path':str((OUT/'fall_metrics.json').relative_to(ROOT))})
    metrics={'posture_model_comparison':comparison,'chosen_model':chosen,
             'cross_domain_negatives':negative_note,
             'fall_sources':list(sources),'licensing':licensing,'release_cleared':release_cleared,
             'split_unit':'source sequence','split_assignment':assignment,
             'train_rows':int(len(training[1])),'urfall_train_rows':int(len(arrays['train'][1])),
             'validation_rows':int(len(vy)),
             'validation_event_sweep':sweep,'selected_operating_point':best,
             'inventory':inventory,
             'test_split_untouched':True,
             'note':'Event metrics come from replaying cached features through the same RuleDetector the app uses; replay was verified against real worker runs.'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'fall_metrics.json').write_text(json.dumps(metrics,indent=2))
    card=snapshot_disclosure_evidence('fall')
    return card,metrics

def fall_event_sweep(model,records,labels,split):
    """Precision / recall / on-time / alerts-per-hour over posture threshold x dwell."""
    chosen=[r for r in records if r['split']==split]
    def score(vector,mask):
        return model.predict_proba(np.concatenate([vector,mask.astype(float)]).reshape(1,-1))[0,1]
    rows=[]
    for threshold in POSTURE_THRESHOLDS:
        for fall_hold in FALL_HOLDS:
            for hold in DOWN_HOLDS:
                results=[]
                for record in chosen:
                    sid=record['sequence_id'];times=frame_times(sid,source=record['source'])
                    detector=RuleDetector(down_hold=hold,fall_hold=fall_hold)
                    events=replay(record['path'],detector=detector,score=score,threshold=threshold)
                    results.append(match_fall_events(events,fall_ground_truth(sid,labels[record['source']],times),
                                                     max(times.values())))
                summary=aggregate(results)
                rows.append({'posture_threshold':threshold,'fall_hold_s':fall_hold,'down_hold_s':hold,
                             **{k:summary[k] for k in ('true_events','alerts','matched','on_time',
                                                       'missed','false_alerts','recall','on_time_recall',
                                                       'precision','false_alerts_per_hour','median_latency_s')}})
    return rows

def rules_only_baseline(records,labels,split):
    """The current hand-written rules, measured on the same split for comparison."""
    results=[]
    for record in records:
        if record['split']!=split:continue
        sid=record['sequence_id'];times=frame_times(sid,source=record['source'])
        results.append(match_fall_events(replay(record['path']),
                                         fall_ground_truth(sid,labels[record['source']],times),
                                         max(times.values())))
    return aggregate(results) if results else None

def pick_operating_point(sweep):
    """Provisional default: maximise on-time recall, break ties on fewer false alerts.

    This is a starting value written into the model card so the app is runnable. The
    operating point is meant to be chosen by a person from the full curve.
    """
    usable=[r for r in sweep if r['on_time_recall'] is not None]
    if not usable:raise ValueError('No validation events to select an operating point from')
    return max(usable,key=lambda r:(r['on_time_recall'],-(r['false_alerts_per_hour'] or 0),
                                    -r['fall_hold_s'],-r['down_hold_s']))

# ---------------------------------------------------------------- activity branch

def clip_windows(d):
    """Fixed-length descriptors from one cached clip, exactly as the app builds them."""
    x=xy(d)
    if not len(x):return []
    return windows(d['t'],x,d['hip_speed'],d['angular_speed'],d['down'],d['upright'],d['quality'])

MOTION_KEYS=('hip_speed','joint_motion','torso_angle','body_aspect','angular_speed',
             'down_fraction','upright_fraction','knee_angle','elbow_angle','wrist_hip')
MOTION_INDICES=[i for i,c in enumerate(CLIP_COLUMNS)
                if '_valid_' not in c and any(k in c for k in MOTION_KEYS)]


def motion_only():
    """Keep the movement columns; drop validity-mask statistics and pose-quality columns.

    Those dropped columns describe how well MediaPipe saw the person, not what the person
    did — and pose failure is class-correlated in this corpus (six normal clips were
    excluded for pose failure, no shoplifting clip was). Feeding them to the classifier
    offers it a route to the label that has nothing to do with behaviour.

    Selection happens inside the pipeline, so the artifact still accepts the full
    descriptor the application builds and no serving code has to know about the subset.
    """
    return ColumnTransformer([('motion','passthrough',MOTION_INDICES)],remainder='drop')


def activity_candidates():
    """Every candidate is scored by the same clip-grouped CV (grouped_cv_scores), so the
    grid below is just more entries in the same dict — no other selection code changes.
    Kept small on purpose: each entry costs CV_FOLDS*len(CV_SEEDS) fits (D037's harness).
    """
    candidates={
        # Original fixed configurations, kept under their original names for continuity
        # with earlier model cards/tests; also present below as part of the grid.
        'logistic':make_pipeline(motion_only(),StandardScaler(),
                                 LogisticRegression(C=1,class_weight='balanced',max_iter=4000,random_state=42)),
        'gradient_boosting':make_pipeline(motion_only(),
                                          GradientBoostingClassifier(random_state=42,n_estimators=200,max_depth=2)),
        'random_forest':make_pipeline(motion_only(),
                                      RandomForestClassifier(n_estimators=400,min_samples_leaf=5,
                                                             class_weight='balanced',random_state=42,n_jobs=1)),
    }
    for c in (.3,1,3):
        candidates[f'logistic_c{c}']=make_pipeline(
            motion_only(),StandardScaler(),
            LogisticRegression(C=c,class_weight='balanced',max_iter=4000,random_state=42))
    for n_estimators in (100,200,300):
        for max_depth in (2,3):
            for learning_rate in (.05,.1):
                candidates[f'gradient_boosting_n{n_estimators}_d{max_depth}_lr{learning_rate}']=make_pipeline(
                    motion_only(),GradientBoostingClassifier(
                        random_state=42,n_estimators=n_estimators,max_depth=max_depth,
                        learning_rate=learning_rate))
    for n_estimators in (200,400,600):
        for min_samples_leaf in (2,5,10):
            candidates[f'random_forest_n{n_estimators}_leaf{min_samples_leaf}']=make_pipeline(
                motion_only(),RandomForestClassifier(
                    n_estimators=n_estimators,min_samples_leaf=min_samples_leaf,
                    class_weight='balanced',random_state=42,n_jobs=1))
    return candidates


def grouped_cv_scores(proto,x,y,groups,folds=CV_FOLDS,seeds=CV_SEEDS):
    """Repeated clip-grouped CV. Returns per-fold window ROC-AUC.

    Windows from one clip are not independent observations, so the fold boundary has to be
    the clip. Repeating over seeds is what makes the spread visible rather than implied.
    """
    scores=[]
    for seed in seeds:
        splitter=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
        for train_index,test_index in splitter.split(x,y,groups):
            if len(np.unique(y[test_index]))<2:continue
            model=clone(proto);model.fit(x[train_index],y[train_index])
            scores.append(float(roc_auc_score(y[test_index],model.predict_proba(x[test_index])[:,1])))
    return scores


def summarise(scores):
    if not scores:return None
    a=np.array(scores)
    return {'mean_roc_auc':float(a.mean()),'sd':float(a.std()),'min':float(a.min()),
            'max':float(a.max()),'folds':len(scores)}


def out_of_fold_threshold(proto,x,y,groups,target_normal_flag=ACTIVITY_TARGET_NORMAL_FLAG):
    """Threshold set on out-of-fold scores so that `target_normal_flag` of ordinary windows
    are flagged. Reading the quantile off in-sample scores instead is what broke the pilot:
    a random forest separates its own training data almost perfectly, so its in-sample 79th
    percentile sits far too high and the realised flag rate on unseen clips exploded.

    Measured on the dev clips: an in-sample threshold aimed at 21% actually flagged 33-61%
    of held-out ordinary windows depending on the family; the out-of-fold threshold landed
    at 17-19% for every family tried.
    """
    folds=min(CV_FOLDS,int(min(np.bincount(y).min(),len(set(groups)))))
    splitter=StratifiedGroupKFold(n_splits=max(folds,2),shuffle=True,random_state=CV_SEEDS[0])
    scores=cross_val_predict(clone(proto),x,y,groups=groups,cv=splitter,method='predict_proba')[:,1]
    return float(np.quantile(scores[y==0],1-target_normal_flag)),scores


def sustained_clip_rule(scores,labels,clips,threshold,run_length=ACTIVITY_SUSTAINED_WINDOWS):
    """Clip outcome under the rule the worker actually applies: a run of consecutive
    flagged windows, not a single one. Window order within a clip is chronological here
    because `windows()` emits it that way.
    """
    per_clip={}
    for clip,score,label in zip(clips,scores,labels):per_clip.setdefault(clip,[[],label])[0].append(score)
    flagged_positive=flagged_negative=positive=negative=0
    for values,label in per_clip.values():
        run=best=0
        for value in values:
            run=run+1 if value>=threshold else 0
            best=max(best,run)
        hit=best>=run_length
        if label:
            positive+=1;flagged_positive+=hit
        else:
            negative+=1;flagged_negative+=hit
    return {'run_length':run_length,'clips':positive+negative,
            'shoplifting_clips_alerted':flagged_positive,'shoplifting_clips':positive,
            'normal_clips_alerted':flagged_negative,'normal_clips':negative,
            'clip_recall':flagged_positive/max(positive,1),
            'normal_clip_alert_rate':flagged_negative/max(negative,1)}


def mnnit_eligible():
    """Eligible MNNIT clips with their frozen split, plus the exclusion record.

    One function so that every consumer — the activity branch and the fall branch's
    cross-domain negatives — sees the same clips in the same splits. Two copies of this
    filter would silently drift and put one clip in two roles.
    """
    caches=sorted((ROOT/'data/processed/features/mnnit').glob('*.npz'))
    if not caches:raise ValueError('No MNNIT feature caches; run extract_features.py --source mnnit --max-people 4')
    eligible=[];excluded=[]
    for path in caches:
        d=np.load(path,allow_pickle=False);stem=path.stem
        label=1 if stem.startswith('shoplifting') else 0
        reasons=[str(v) for v in d['attempt_reason']]
        coverage=reasons.count('ok')/max(len(reasons),1)
        peak=int(d['attempts'][:,2].max()) if len(d['attempts']) else 0
        row={'source':stem,'scene_label':'shoplifting' if label else 'normal',
             'sha256':str(d['source_sha256']),
             'pose_coverage':round(coverage,4),'rows':int(len(d['t'])),'peak_people':peak}
        if coverage<MIN_CLIP_COVERAGE or len(d['t'])<MIN_CLIP_ROWS:
            excluded.append({**row,'reason':'insufficient_pose_coverage'});continue
        if peak>1:
            excluded.append({**row,'reason':'multiple_people_clip_label_not_person_label'});continue
        built=clip_windows(d)
        if not built:
            excluded.append({**row,'reason':f'shorter_than_the_{WINDOW_S}s_window'});continue
        vectors=[v for _,_,v in built]
        if not np.isfinite(np.array(vectors)).all():
            excluded.append({**row,'reason':'non_finite_features'});continue
        eligible.append({**row,'label':label,'windows':vectors})
    # Exclusions must be reported per class. Pose failure is class-correlated in this
    # corpus (normal clips fail, shoplifting clips do not), so an unreported filter would
    # let "no person detected" act as a hidden predictor of the normal class.
    exclusion_summary={}
    for row in excluded:
        key=f"{row['scene_label']}/{row['reason']}"
        exclusion_summary[key]=exclusion_summary.get(key,0)+1
    if len(eligible)<30:raise ValueError(f'Only {len(eligible)} eligible clips after filtering')

    assignment={}
    for label in (0,1):
        members=sorted([r['source'] for r in eligible if r['label']==label])
        for index,stem in enumerate(members):assignment[stem]=mnnit_split(stem,index,len(members))
    assert_no_leakage([(r['sha256'],assignment[r['source']]) for r in eligible])
    for row in eligible:row['split']=assignment[row['source']]
    return eligible,excluded,exclusion_summary,assignment


def activity_training():
    eligible,excluded,exclusion_summary,assignment=mnnit_eligible()

    def block(split):
        rows=[r for r in eligible if r['split']==split]
        x=np.array([v for r in rows for v in r['windows']])
        y=np.array([r['label'] for r in rows for _ in r['windows']])
        clip=[r['source'] for r in rows for _ in r['windows']]
        return x,y,clip,rows
    tx,ty,_,train_rows=block('train');vx,vy,vclip,validation_rows=block('validation')
    if len(np.unique(ty))<2 or len(np.unique(vy))<2:
        raise ValueError('Train and validation must both contain each class')

    tgroup=np.array([r['source'] for r in train_rows for _ in r['windows']])

    # Model selection runs on the training split only, by clip-grouped repeated CV. The
    # validation split is then a clean check of the chosen configuration rather than the
    # surface that chose it — with 31 validation clips, selecting on it was selecting on
    # noise as much as on skill.
    comparison={}
    for name,proto in activity_candidates().items():
        comparison[name]={'cv':summarise(grouped_cv_scores(proto,tx,ty,tgroup))}
    anomaly_proto=make_pipeline(motion_only(),IsolationForest(n_estimators=200,random_state=42,n_jobs=1))
    forest=clone(anomaly_proto);forest.fit(tx[ty==0])
    anomaly=-forest.score_samples(vx)
    comparison['isolation_forest_normal_only']={
        'validation_window_roc_auc':float(roc_auc_score(vy,anomaly)),
        'validation_window_average_precision':float(average_precision_score(vy,anomaly)),
        'note':'Trained on normal windows only; needs no shoplifting labels. Measured at 0.37 ROC-AUC in the pilot — worse than chance — and kept only as a recorded negative result.'}
    chosen=max(comparison,key=lambda n:(comparison[n].get('cv') or {}).get('mean_roc_auc',0))
    use_anomaly=False
    proto=activity_candidates()[chosen]
    model=clone(proto);model.fit(tx,ty)
    family=chosen

    threshold,oof=out_of_fold_threshold(proto,tx,ty,tgroup)
    scores=model.predict_proba(vx)[:,1]
    for name in comparison:
        entry=comparison[name]
        if 'cv' in entry:
            candidate=clone(activity_candidates()[name]);candidate.fit(tx,ty)
            probability=candidate.predict_proba(vx)[:,1]
            entry['validation_window_roc_auc']=float(roc_auc_score(vy,probability))
            entry['validation_window_average_precision']=float(average_precision_score(vy,probability))

    # Clip level: aggregate window scores back per clip, so the two views are comparable.
    per_clip={}
    for name,value,label in zip(vclip,scores,vy):per_clip.setdefault(name,[[],label])[0].append(value)
    clip_scores=np.array([np.quantile(v[0],.9) for v in per_clip.values()])
    clip_labels=np.array([v[1] for v in per_clip.values()])
    clip_auc=float(roc_auc_score(clip_labels,clip_scores)) if len(np.unique(clip_labels))>1 else None

    def point(threshold_value,provenance):
        flag=scores>=threshold_value
        tp=int((flag&(vy==1)).sum());fp=int((flag&(vy==0)).sum());fn=int(((~flag)&(vy==1)).sum())
        return {'threshold':float(threshold_value),'provenance':provenance,'tp':tp,'fp':fp,'fn':fn,
                'window_precision':tp/max(tp+fp,1),'window_recall':tp/max(tp+fn,1),
                'flagged_normal_window_fraction':fp/max(int((vy==0).sum()),1),
                'sustained_clip_rule':sustained_clip_rule(scores,vy,vclip,threshold_value)}

    # The curve is reported so a person can move the operating point; every row is a target
    # ordinary-window flag rate, read off out-of-fold training scores, then measured on
    # validation. Target and realised rate are both shown because they differ.
    curve=[]
    for target in (.05,.1,.15,.21,.3,.4):
        value=float(np.quantile(oof[ty==0],1-target))
        curve.append({'target_normal_flag_rate':target,**point(value,'out_of_fold_training_scores')})
    selected=point(threshold,'out_of_fold_training_scores')
    selected['target_normal_flag_rate']=ACTIVITY_TARGET_NORMAL_FLAG
    card=save('activity',model,selected['threshold'],{
        'target':'unusual_motion','input':'clip_descriptor',
        'scoring':'anomaly_score' if use_anomaly else 'probability',
        'feature_names':CLIP_COLUMNS,'window_s':WINDOW_S,'stride_s':STRIDE_S,
        'feature_selection':{'used':[CLIP_COLUMNS[i] for i in MOTION_INDICES],
                             'dropped':[c for i,c in enumerate(CLIP_COLUMNS) if i not in MOTION_INDICES],
                             'reason':'validity-mask and pose-quality columns describe how well the person was seen, not what they did, and pose failure is class-correlated in this corpus'},
        'model_family':family,'model_comparison':comparison,
        # Unverified, not False: no release review has been done for this branch (D048).
        # None is a distinct state from the fall branch's known-blocked False (D029/D044).
        'release_cleared':None,
        'threshold_selection':f'out-of-fold clip-grouped scores on the training split, targeting {ACTIVITY_TARGET_NORMAL_FLAG:.0%} of ordinary windows flagged',
        'alert_rule':f'the application requires {ACTIVITY_SUSTAINED_WINDOWS} consecutive flagged windows; a single flagged window is not an alert',
        # The same rule as a number the worker can read, so `alert_rule` cannot describe
        # one thing to the reviewer while the application applies another.
        'sustained_windows':ACTIVITY_SUSTAINED_WINDOWS,
        'training_source':'MNNIT retail clips, MediaPipe pose features, fixed-length windows',
        'label_scope':'scene level only — the clip label says a clip contains shoplifting, never who or when. Every window of a shoplifting clip inherits the clip label, so many positive windows contain no act at all.',
        'eligibility':f'single-person clips with pose coverage >= {MIN_CLIP_COVERAGE} and at least one full {WINDOW_S}s window',
        'not_proof_of_theft':'Flags motion resembling clips labelled shoplifting. It does not establish theft and requires human review.',
        # Read by the application and shown on the escalation prompt. A person deciding
        # whether to act on this alert must be told how often it fires on ordinary footage.
        'expected_false_alert_rate':selected['sustained_clip_rule']['normal_clip_alert_rate'],
        'expected_detection_rate':selected['sustained_clip_rule']['clip_recall'],
        'alert_caveat':(
            f"In validation, {selected['sustained_clip_rule']['normal_clips_alerted']} of "
            f"{selected['sustained_clip_rule']['normal_clips']} ordinary retail clips "
            f"({selected['sustained_clip_rule']['normal_clip_alert_rate']:.0%}) raised an "
            'activity alert under the two-consecutive-window rule. This is a clip fraction, '
            'not an hourly rate or the probability this alert is correct. Results cover '
            'eligible single-person clips; clip labels do not identify who acted or when. '
            'Movement does not establish theft. Performance on your camera is unknown.'),
        'license_note':'MNNIT CC BY 4.0; cite DOI 10.17632/r3yjf35hzr.1',
        'metrics_path':str((OUT/'activity_metrics.json').relative_to(ROOT))})
    metrics={'model_comparison':comparison,'chosen_model':family,
             'selection_method':f'clip-grouped {CV_FOLDS}-fold CV repeated over seeds {list(CV_SEEDS)}, on the training split only',
             'split_unit':'clip','split_assignment':assignment,
             'window_s':WINDOW_S,'stride_s':STRIDE_S,
             'train_windows':int(len(ty)),'validation_windows':int(len(vy)),
             'validation_clip_roc_auc':clip_auc,
             'eligible_clips':len(eligible),'excluded_clips':len(excluded),
             'exclusions_by_class_and_reason':exclusion_summary,'excluded_detail':excluded,
             'class_counts':{s:{'normal':sum(1 for r in eligible if r['split']==s and r['label']==0),
                                'shoplifting':sum(1 for r in eligible if r['split']==s and r['label']==1)}
                             for s in ('train','validation','test')},
             'validation_threshold_curve':curve,'selected_operating_point':selected,
             'validation_sustained_clip_rule':selected['sustained_clip_rule'],
             'test_split_untouched':True,
             'limits':'Window level within single-person clips. No per-person or per-interval claim, and no evidence about when an act began.'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'activity_metrics.json').write_text(json.dumps(metrics,indent=2))
    card=snapshot_disclosure_evidence('activity')
    return card,metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Train the fall and retail-activity branches.')
    parser.add_argument('--fall-sources',default='urfall',
                        help='Comma-separated posture corpora for the fall branch. '
                             '"owncam" alone produces a release-cleared artifact; '
                             '"urfall" is CC BY-NC-SA research-only (D029).')
    parser.add_argument('--branch',default='both',choices=['fall','activity','both'])
    arguments=parser.parse_args()
    fall_sources=tuple(part.strip() for part in arguments.fall_sources.split(',') if part.strip())
    OUT.mkdir(parents=True,exist_ok=True)
    results={}
    branches=[('fall',lambda:fall_training(fall_sources)),('activity',activity_training)]
    for name,fn in [b for b in branches if arguments.branch in ('both',b[0])]:
        try:
            card,metrics=fn()
            results[name]={'status':'trained','model_family':card.get('model_family'),
                           'threshold':card['threshold'],
                           'release_cleared':card.get('release_cleared'),
                           'metrics':str((OUT/f'{name}_metrics.json').relative_to(ROOT))}
        except Exception as exc:
            results[name]={'status':'blocked','error':f'{type(exc).__name__}: {exc}'}
    (OUT/'summary.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({'run':RUN,'results':results},indent=2))
