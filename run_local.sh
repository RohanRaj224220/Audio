#!/usr/bin/env bash
# Start the app on this computer only (not visible to other devices on the network).
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
exec streamlit run streamlit_app.py --server.address=localhost "$@"
