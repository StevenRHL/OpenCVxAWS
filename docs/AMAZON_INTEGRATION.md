# Amazon (AWS) integration — proposal

Date: 2026-09-14. Owner: Steven. Status: planning only — not implemented, not authorized to
run paid cloud jobs, upload footage, or create AWS resources. This document is a proposal to
review and decide on later, not a build order.

## Why

The app already has a **local learning-candidate queue** (`watchverify/review.py`,
`watchverify/ui/queue.py`): a reviewer can flag a missed or wrong event in the debugger, correct
its label, mark it `ready_for_dataset_review`, and export it. `review.export_candidates()`
copies those clips plus a `manifest.json` into `outputs/_exports/<id>/` — but the code is
explicit that this is `# --- export (dataset review, not training) ---`. It stages labeled data
for a human; nothing is uploaded anywhere, and nothing retrains automatically.

This proposal asks whether that local staging step could be extended, opt-in, so that:

1. reviewed clips can be archived off-machine (S3), so a collaborator without the original
   training data can still contribute new labeled examples without you manually copying files
   around, and
2. an AWS ML service could run a *second opinion* over those clips, as an extra signal a human
   reviewer sees alongside the local model's output — not as a replacement for local training.

## What exists today (and what this must not contradict)

- **No AWS integration exists in this repo right now.** The only AWS-adjacent strings anywhere
  are an S3 download URL for the MNNIT dataset archive (`data/manifests/downloads.jsonl`) and one
  unrelated backlog line in `docs/IDEAS.md` about AWS Arm *compute* for a competition track —
  neither is storage or ML integration. There is no `boto3`, no credentials plumbing, nothing.
- **Storage today**: each run lives at `outputs/<run_id>/`; the source video is copied to
  `outputs/<run_id>/source.<ext>` and made read-only. Promoted clips archive to
  `outputs/_retained_learning/<run_id>/clips/`. There is no background task queue — `jobs.py`
  runs one worker subprocess at a time and refuses a second concurrent analysis. Any upload step
  needs its own async mechanism; there's nothing to piggyback on today.
- **Constraints this proposal must respect, quoted exactly:**
  - `README.md:36` — "Processing is local. Annotated exports are silent."
  - `AGENTS.md:7` — "Do not treat this document as authorization to run paid cloud jobs, publish
    footage, send messages, or spawn other agents."
  - `.gitignore`'s credentials section — "Nothing in this project needs a secret to run; these
    patterns exist so that an accidentally created one is never committed."

  Any AWS feature touches all three. It must be opt-in, off by default, and explicitly
  authorized before anyone connects real AWS credentials or footage to it.

## Proposed extension (opt-in, cost-conscious)

- **Scope of what's uploaded**: only clips that have already been through local review and
  marked `ready_for_dataset_review` — i.e. only what `export_candidates()` already produces.
  Never raw, unreviewed video. Never automatic.

- **Two-layer opt-in gate, to control cost**: a config-level toggle (off by default) *plus* a
  separate per-batch confirmation before any upload or AWS ML call actually fires. Turning the
  feature on once should never be enough, by itself, to cause a future export to silently incur
  charges — each batch needs its own explicit go-ahead. This is the main answer to "is this
  worth the cost": nothing runs, and nothing bills, without a deliberate action every time.

- **AWS ML role — recommended as an independent secondary analysis, not a training dependency.**
  For example, Rekognition Video/Image label or person/pose detection run over an exported
  clip's frames, with results shown next to the local model's own output in the debugger, purely
  for a human to compare. It would **not** feed `scripts/train_models.py` automatically —
  that stays a manual, reviewed step, the same way `export_candidates()` already draws that line
  today. This keeps the (expensive) cloud call optional and bounded, rather than a dependency of
  the core pipeline.

- **Cost controls to work out before building anything**: a batch size cap (the review queue
  already caps exports at `MAX_EXPORT_BATCH = 200` in `review.py` — a cloud batch should likely
  be much smaller), a cost estimate shown before confirming a batch, and a per-run or per-month
  spend ceiling.

- **Explicitly out of scope for this document** (open questions, not decisions): IAM credential
  storage/rotation, consent mechanics for footage of identifiable people, S3 retention/deletion
  policy, encryption-at-rest specifics, and whether this ever becomes default-on. None of these
  are resolved here.

## Sketch of the flow

```
review candidate (existing, local)
  -> export_candidates()                          [existing, local, unchanged]
  -> [proposed, opt-in] upload exported bundle to S3
  -> [proposed, opt-in] run AWS ML pass over the clip's frames
  -> results attached to the export bundle for a human to read
  -> (still manual) scripts/import_reviewed_exports.py folds it into an owncam-style
     local 'reviewed' source, then scripts/prepare_owncam.py --corpus reviewed,
     scripts/extract_features.py --source reviewed
  -> (still manual) scripts/train_models.py --fall-sources urfall,reviewed retrain
```

The last two steps are no longer hypothetical — `scripts/import_reviewed_exports.py` is the
local, offline version of that hand-off (no AWS needed for it), added alongside this proposal.
Everything above it in the sketch (S3 upload, AWS ML pass) is still just proposed.

Every "proposed" step requires the two-layer opt-in gate above. Every step after it is exactly
what already happens today.

## Status

This document does not authorize creating AWS resources, incurring any cost, or uploading any
footage. It is a proposal for review and a future explicit decision.
