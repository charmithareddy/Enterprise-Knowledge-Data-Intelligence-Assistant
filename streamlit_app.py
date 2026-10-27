"""
Main application entrypoint for Streamlit Community Cloud and local deployment.
Seamlessly delegates execution to app/streamlit_app.py.
"""

from pathlib import Path
import runpy
import sys

# Ensure root directory is on Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

APP_TARGET = ROOT_DIR / "app" / "streamlit_app.py"
runpy.run_path(str(APP_TARGET), run_name="__main__")
