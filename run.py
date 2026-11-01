"""
Root Entry Point for the Major Project
Delegates to ml_preprocessing/run_pipeline.py
"""
import os
import subprocess
import sys

if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(__file__), "ml_preprocessing", "run_pipeline.py")
    sys.exit(subprocess.call([sys.executable, script_path]))

