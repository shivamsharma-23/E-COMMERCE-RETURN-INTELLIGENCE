"""
Streamlit Community Cloud Entry Point Wrapper.
Ensures seamless deployment whether Main File Path is configured as 'app.py' or 'streamlit_app.py'.
"""

import os
import sys
import runpy

# Ensure current directory is on python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

if __name__ == "__main__":
    runpy.run_path(os.path.join(current_dir, "app.py"), run_name="__main__")
