import os
import subprocess
import sys

#Set the working directory
BASE_DIR = "enter your path here"
os.chdir(BASE_DIR)

#Automatically launch Streamlit user interface
script_path = os.path.join(BASE_DIR, "script_py.py")
subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])