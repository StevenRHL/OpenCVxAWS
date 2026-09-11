#!/bin/zsh
cd -- "$(dirname -- "$0")" || exit 1
export MPLCONFIGDIR="$PWD/.runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR"
if [[ ! -x .venv/bin/python ]]; then
  print 'Please double-click Setup WatchVerify.command first.'
  read 'reply?Press Enter to close.'
  exit 1
fi
print 'Opening the local video review app. Leave this window open while using it.'
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
