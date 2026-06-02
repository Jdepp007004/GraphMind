"""
scripts/run_dashboard.py

Entry point: launches Streamlit dashboard via subprocess.
"""

import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    """Launch the Streamlit dashboard."""
    dashboard_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "dashboard", "app.py"
    )
    subprocess.run(["streamlit", "run", dashboard_path, "--server.port", "8501"])
