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

A model comparison on 11 September 2026 produced two challengers. Neither was installed,
and the reasons are recorded below under "Why the new models were not installed".

Full browser upload/review/decision/cancel/restart/seek/export validation and clean setup
remain open. Representative camera footage and new held-out evaluation are needed before
further model claims. Own-camera preparation/capture scripts exist, but no own-camera
corpus is present. The app accepts files; live-camera inference is a later milestone.

70 UR Fall sequences and 179 MNNIT clips have been prepared according to the recorded
release report. Source inventories and use restrictions are in `data/manifests/` and
`docs/DATASETS.md`. RetailS remains archived; full PoseLift/UCF acquisition is unfinished.

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
and cards. `scripts/`: preparation/training tools. `outputs/`: saved analyses.
`docs/STATUS.md`: latest handoff followed by historical notes. `docs/COUNCIL_REVIEW.md`:
architecture constraints. `CONTINUE_HERE.md`: next bounded part and manual instructions.
