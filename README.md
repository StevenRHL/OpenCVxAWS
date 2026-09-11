# WatchVerify — experimental video review

The VSC folder is the current project. Start with `CONTINUE_HERE.md` for the latest
completed part, checks, and next step. The JKO/opencv-ai-build copy is older.

## Open it

Double-click **Launch WatchVerify.command**. Keep its terminal open and use
http://127.0.0.1:8501. The existing environment works on this Mac; a clean-install test
remains pending. If dependencies are missing, run **Setup WatchVerify.command** first.

Upload a short recording, optionally supply its known recording time with UTC offset,
and select **Start analysis**. Review the timestamped timeline, save relevant/false-alarm/
unclear labels, and download annotated MP4 or JSON/CSV. Limits: ten minutes, 500 MB,
1920×1080. Processing is local. Annotated exports are silent.

## Current capabilities

- MediaPipe body pose, tracking, past-only features, trained fall/activity classifiers,
  experimental rules, immutable initial alerts and incident revisions.
- Local run management, cancellation, review persistence and video/event exports.
- Independent police and medical decision records. Buttons record a decision; the app
  contacts nobody. New runs snapshot the loaded models' disclosures so later model
  changes do not change the figures attached to a run. Older runs without snapshots
  explicitly show that model-specific evaluation information was not saved.
- Completed RGB research-video runs and focused software tests. See `docs/EXPERIMENTS.md`
  for exactly what was checked; these checks do not establish detection reliability.

## Model evidence and limits

Installed models are from `pilot-20260910T232430Z`. The activity validation recorded alerts
on 3/15 ordinary clips (20%) under the two-consecutive-window rule. Fall validation
recorded 11 alerts: 6 matched labelled falls and 5 were unmatched. These are small
validation samples, not reliable hourly rates or a probability that a particular alert
is correct. They were used during development. Separate test results and limitations
are in `docs/RELEASE_REPORT.md`; Part 2 did not rerun accuracy evaluation.

Activity labels apply to clips, not a particular person/action interval. Movement does
not establish theft. Posture does not diagnose injury. Empty alerts do not establish safety.
Release permission for the fall artifact remains unresolved in project records; the
activity artifact has no completed release-clearance review recorded. Model cards state
these limits explicitly. No new legal clearance is claimed.

## Remaining work

Full browser upload/review/decision/cancel/restart/seek/export validation and clean setup
remain open. Representative camera footage and new held-out evaluation are needed before
further model claims. Own-camera preparation/capture scripts exist, but no own-camera
corpus is present. The app accepts files; live-camera inference is a later milestone.

70 UR Fall sequences and 179 MNNIT clips have been prepared according to the recorded
release report. Source inventories and use restrictions are in `data/manifests/` and
`docs/DATASETS.md`. RetailS remains archived; full PoseLift/UCF acquisition is unfinished.

## Project files

`app.py`: interface. `watchverify/`: processing and storage. `models/`: installed artifacts
and cards. `scripts/`: preparation/training tools. `outputs/`: saved analyses.
`docs/STATUS.md`: latest handoff followed by historical notes. `docs/COUNCIL_REVIEW.md`:
architecture constraints. `CONTINUE_HERE.md`: next bounded part and manual instructions.
