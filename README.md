# WatchVerify — experimental video review

[![CI](https://github.com/StevenRHL/OpenCVxAWS/actions/workflows/ci.yml/badge.svg)](https://github.com/StevenRHL/OpenCVxAWS/actions/workflows/ci.yml)

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
the installed model. It also exposed a limitation that affected the installed model too —
**a fall could be missed entirely if body pose dropped out between the fall itself and the
landing.** That limitation has since been fixed; see "The pose-dropout fall miss" below.

Any sequence can be inspected the same way:

```sh
.venv/bin/python scripts/explain_fall_event.py --sequence fall-01
```

It prints the per-frame posture score against the threshold, the tracked identity, the
rule state, and a plain-language reason for the alert or the silence.

## The pose-dropout fall miss, and the fix

The limitation above is closed. Fall evidence no longer belongs solely to the tracked
identity that recorded it: when an identity ends while still holding an unspent
rapid-posture-change marker, `EvidenceRelay` (in `watchverify/core.py`) offers that marker
to the identity the tracker issues in its place.

The relay widens nothing. The marker keeps its original timestamp, so the same
`transition_window` that governs a fall on one unbroken identity governs a carried one —
evidence that would have expired stays expired — and the alert still needs a down posture
of its own. A replacement more than two torso lengths from where the old identity vanished
is treated as a different person and refused, and evidence is consumed by the first
identity that takes it, so one fall cannot seed alerts on several people. Model artifacts
and thresholds are unchanged.

An alert assembled this way carries the observation
`evidence_carried_across_identity_change`, and the run reports `carried_fall_evidence` in
its metrics, because a reviewer opening that clip will find a visible break between the
movement and the landing and should be told to expect it.

**What it changes, measured.** Changing detection rules invalidates the figures they were
measured under, so the whole corpus was re-measured at the same threshold, dwell constants
and splits. The numbers are in `models/fall.json` under `revalidation`, and summarised here:

| | before | after |
|---|---|---|
| Validation: labelled falls matched / alerts | 6 of 6 / 11 | 6 of 6 / 12 |
| Test: labelled falls matched / alerts | 6 of 6 / 9 | 6 of 6 / 10 |
| All 30 fall sequences: no alert of any kind | 3 | 1 |
| All 30 fall sequences: reported by the fast path | 27 | 29 |
| 40 ADL sequences (fall negatives): alerts | 21 | 23 |

Recall did not fall anywhere measured. Two labelled falls that previously produced no alert
at all — `fall-13` and `fall-19` — are now reported, at a cost of two additional alerts
across the forty fall-negative sequences. By the rule already recorded above for the model
comparison — fewer false alarms is not worth missing more of what we are looking for —
that trade is taken. `fall-25` is still silent, but for an unrelated reason: the model never
calls a frame down on it at all (peak 0.053 against a 0.3 threshold), which is a posture
-scoring miss and not an evidence gap.

**Two caveats on where those numbers come from.** The whole-corpus row includes the training
split, and is a diagnostic of the failure mode rather than an accuracy claim — the held-out
splits happen to contain no sequence that ends before the two-second person-down fallback
completes, which is the only situation in which this failure is fatal rather than merely
slow. And the re-measurement runs through `evaluation.replay`, over feature caches whose
identities were assigned at extraction time. Running the same UR Fall clips through the
worker directly did not trigger the relay at all: on that path the pose dropouts are shorter
than the one second the tracker will hold an identity across, so no renumbering occurs and
there is nothing to carry. The relay is therefore a safety net on the worker path for this
corpus, not a measured improvement to it. That the net is actually connected is pinned
end-to-end by `tests/test_worker_evidence_relay.py`, which drives the real tracker through a
blind span long enough to end an identity and asserts that the worker reports nothing
without the relay and a labelled `possible_fall` with it.

## Live camera preview

**Live camera** in the sidebar opens a camera and runs the same detection stack on it, so
you can see whether a given camera, height and angle produce usable body pose before
recording with it. It is a preview only: nothing is written to disk, no analysis is created,
and nothing reaches the admin dashboard. An empty timeline from a camera that never resolved
a body is not evidence of a quiet room. Recorded analysis still runs on uploaded files.

## Admin dashboard

**Admin dashboard** collects every observation awaiting a decision across every analysis
into one queue, and cuts each one out of its recording so the clip opens on the incident
instead of on minute zero — five seconds before the event by default, since a clip that
starts on the alert shows the consequence and not the cause. Deciding there writes to the
same append-only escalation log as the per-analysis prompt, so the two views cannot disagree
about what was decided. It contacts nobody.

## Project files

`app.py`: the upload-and-review interface. `pages/`: the admin dashboard and the live
camera preview, reached from the sidebar. `watchverify/`: processing and storage —
`core.py` (geometry, tracking, rules, the evidence relay), `worker.py` (one analysis run),
`incidents.py` (the cross-run queue and per-incident clips), `live.py` (camera preview),
`ui.py` (the look shared by every page). `models/`: installed artifacts and cards.
`scripts/`: acquisition, preparation, training and verification tools.
`data/manifests/`: what was downloaded, prepared and split. `outputs/`: saved analyses.
Handoff, architecture and decision notes live with the build machine's working documents.
