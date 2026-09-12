# WatchVerify — experimental video review

The VSC folder is the current project; the JKO/opencv-ai-build copy is older. Planning,
status and handoff notes are kept on the build machine and are not published here, so this
file and `HOW TO RUN.txt` are what a checkout has.

## Install it

Double-click **Setup WatchVerify.command** once. It needs Python 3.10-3.12 and reports
the versions it found if none fits, installs `requirements-lock.txt`, downloads the
MediaPipe pose landmarker (about 15 MB, `models/*.task`, not in the repository) against a
pinned checksum from an immutable upstream revision, and then runs one real inference on a
synthetic frame. It prints PASS only if the installation can actually analyse something.

`scripts/acquire_pose_assets.py` fetches the asset on its own, and
`scripts/verify_install.py` runs the smoke check on its own. Every fetch appends its
origin, revision, size, checksum and licence to `data/manifests/downloads.jsonl`, which is
why that file shows as modified after setup.

## Open it

Double-click **Launch WatchVerify.command**. Keep its terminal open and use
http://127.0.0.1:8501.

A clean install was tested on 12 September 2026 by cloning this repository into an empty
directory: setup selected Python 3.10.0, installed the 73 pinned packages, downloaded and
checksum-verified both pose variants, and passed its smoke inference. `pytest tests/` in
that clone reported 177 passed, 13 skipped, 0 failed; each skip names the prepared footage,
training run or feature cache that is deliberately not in the repository. In the app,
**Start analysis** was enabled, a 5.3-second UR Fall clip produced one possible-fall
observation at 00:00:03.5, and the exported annotated MP4 decoded to all 160 frames.

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
- Completed RGB research-video runs and focused software tests. Exactly what was checked is
  recorded in the build machine's experiment log; these checks do not establish detection
  reliability.

## Model evidence and limits

Installed models are from `pilot-20260910T232430Z`. The activity validation recorded alerts
on 3/15 ordinary clips (20%) under the two-consecutive-window rule. Fall validation
recorded 11 alerts: 6 matched labelled falls and 5 were unmatched. These are small
validation samples, not reliable hourly rates or a probability that a particular alert
is correct. They were used during development. Separate test results and limitations are
in the release report kept with the project notes; Part 2 did not rerun accuracy evaluation.

Activity labels apply to clips, not a particular person/action interval. Movement does
not establish theft. Posture does not diagnose injury. Empty alerts do not establish safety.
Release permission for the fall artifact remains unresolved in project records; the
activity artifact has no completed release-clearance review recorded. Model cards state
these limits explicitly. No new legal clearance is claimed.

## Remaining work

A model comparison on 11 September 2026 produced two challengers. Neither was installed,
and the reasons are recorded below under "Why the new models were not installed".

Clean setup is now tested from a fresh clone, as described above. Browser validation of
review, decision, cancel, restart and seek remains open; only upload, analysis and export
were exercised. Representative camera footage and new held-out evaluation are needed before
further model claims. Own-camera preparation/capture scripts exist, but no own-camera
corpus is present. The app accepts files; live-camera inference is a later milestone.

70 UR Fall sequences and 179 MNNIT clips have been prepared according to the recorded
release report. Source inventories and use restrictions are in `data/manifests/`.
RetailS remains archived; full PoseLift/UCF acquisition is unfinished.

## Why the new models were not installed

A bounded comparison tried two new model families against the installed ones. Both were
rejected and the installed models are unchanged.

The **activity challenger** raised fewer alerts on ordinary clips (2 of 15 instead of 3 of
15) but also detected one fewer of the labelled clips (12 of 16 instead of 13 of 16).
Fewer false alarms is not worth missing more of what we are looking for, so it was kept as
an experiment only.

The **fall challenger** looked better on paper — the same 6 of 6 labelled falls detected on
time, with 4 unmatched alerts instead of 5 — but it reported nothing at all on `fall-01`,
a fall the installed model does alert on. That was investigated rather than explained away,
and the cause turned out to be worth knowing:

1. The person falls. The detector records a "rapid posture change" marker at 3.703s.
2. Body pose is then lost for 0.4 seconds, exactly while the person is landing.
3. The person is picked up again at 4.204s as a *new* person, because the tracker cannot
   confidently match someone across the gap. The marker belonged to the old identity and
   is discarded with it.
4. The new identity sees only someone already lying still, which looks the same as someone
   who lay down on purpose. Without the marker, no fall alert can be raised.
5. The person never stands up again in the clip, so the marker can never be re-recorded,
   and the backup rule — alerting after two seconds of lying still — needs longer than the
   clip has left.

The installed model escapes this only by luck. Its threshold is lower (0.3 against 0.8), so
it called the person down at 3.603s, one frame *before* the pose was lost, while the marker
was still alive. The challenger is not worse at recognising a person on the ground; it is
actually more confident once they are down. It simply crosses its threshold a moment later,
and that moment falls inside the blind gap.

So this is a real miss, not a measurement artefact, and the documented rule applies: keep
the installed model. It also exposes a limitation that affects the installed model too —
**a fall can be missed entirely if body pose drops out between the fall itself and the
landing.** That is now covered by `tests/test_fall_evidence_gap.py` so the behaviour cannot
change unnoticed. The detection rules were deliberately left alone: changing them would
invalidate the validation figures quoted above, which would have to be re-measured first.

Any sequence can be inspected the same way:

```sh
.venv/bin/python scripts/explain_fall_event.py --sequence fall-01
```

It prints the per-frame posture score against the threshold, the tracked identity, the
rule state, and a plain-language reason for the alert or the silence.

## Project files

`app.py`: interface. `watchverify/`: processing and storage. `models/`: installed artifacts
and cards. `scripts/`: acquisition, preparation, training and verification tools.
`data/manifests/`: what was downloaded, prepared and split. `outputs/`: saved analyses.
Handoff, architecture and decision notes live with the build machine's working documents.
