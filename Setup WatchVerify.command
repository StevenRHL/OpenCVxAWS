#!/bin/zsh
# Install and then PROVE the installation can run.
#
# pip finishing successfully is not evidence that WatchVerify works: the MediaPipe pose
# asset is not part of the repository, and without it the app loads, disables Start, and
# cannot analyse anything. So this script pins the environment, fetches the asset, and
# ends by running one real inference.
cd -- "$(dirname -- "$0")" || exit 1

# Supported interpreter range, determined from the pins in requirements-lock.txt:
#   floor   3.10 - av, scipy, scikit-learn, matplotlib, jax and others declare
#                  Requires-Python >=3.10
#   ceiling 3.12 - mediapipe 0.10.21 and numpy 1.26.4 publish no wheels above cp312
# Both bounds are hard: outside them the pinned set cannot be installed at all.
MIN_MINOR=10
MAX_MINOR=12

fail() {
  print -u2 "\nSetup did not finish: $1"
  read 'reply?Press Enter to close.'
  exit 1
}

interpreter_minor() {
  "$1" -c 'import sys; print(sys.version_info[0]*1000+sys.version_info[1]) if sys.version_info[0]==3 else print(0)' 2>/dev/null
}

# Pick an interpreter inside the supported range, reporting every version actually found.
PYTHON=''
found=''
for candidate in ${PYTHON_BIN:-} python3 python3.12 python3.11 python3.10; do
  [[ -z $candidate ]] && continue
  whence -p -- "$candidate" >/dev/null 2>&1 || continue
  version=$("$candidate" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])' 2>/dev/null) || continue
  code=$(interpreter_minor "$candidate")
  found="$found  $candidate -> Python $version\n"
  if [[ -n $code && $code -ge $((3000+MIN_MINOR)) && $code -le $((3000+MAX_MINOR)) ]]; then
    PYTHON="$candidate"
    print "Using $candidate (Python $version)."
    break
  fi
done

if [[ -z $PYTHON ]]; then
  print -u2 "WatchVerify needs Python 3.$MIN_MINOR to 3.$MAX_MINOR. Versions found on this Mac:"
  if [[ -n $found ]]; then print -u2 -- "$found"; else print -u2 '  no python3 on PATH'; fi
  fail "no supported Python was found. Install Python 3.12 from python.org, then run this again."
fi

# An existing .venv built on an unsupported interpreter is reported, not silently deleted.
if [[ -x .venv/bin/python ]]; then
  code=$(interpreter_minor .venv/bin/python)
  version=$(.venv/bin/python -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])' 2>/dev/null)
  if [[ -z $code || $code -lt $((3000+MIN_MINOR)) || $code -gt $((3000+MAX_MINOR)) ]]; then
    fail ".venv already exists and runs Python ${version:-unknown}, outside 3.$MIN_MINOR-3.$MAX_MINOR. Delete the .venv folder and run this again."
  fi
else
  print 'Creating the .venv environment...'
  "$PYTHON" -m venv .venv || fail 'could not create the .venv environment.'
fi

# requirements-lock.txt, not requirements.txt: the lock file pins all 73 packages that
# were present when the installed models were built. requirements.txt names only the 11
# direct ones and leaves their dependencies free to resolve to anything.
print 'Installing pinned packages (this can take several minutes)...'
.venv/bin/python -m pip install --disable-pip-version-check -r requirements-lock.txt \
  || fail 'the pinned packages could not be installed. Check your internet connection and run this again.'

print 'Fetching the pose model asset...'
.venv/bin/python scripts/acquire_pose_assets.py \
  || fail 'the pose model asset could not be fetched or did not match its expected checksum (details above). WatchVerify cannot analyse video without it.'

print 'Checking that this installation can actually run inference...'
.venv/bin/python scripts/verify_install.py \
  || fail 'the installation could not run inference (details above).'

print '\nSetup finished and verified. Double-click Launch WatchVerify.command.'
# The prompt is for the double-click case; it must not decide the exit status, which
# `read` would otherwise set to 1 whenever this runs without a terminal.
read 'reply?Press Enter to close.'
exit 0
