import os
import subprocess
import sys

# Add project root to Python path so you can import the road_accident_risk package
sys.path.append(os.getcwd())

# Move into the Streamlit app folder
os.chdir("app")

# Start Streamlit (Render passes its own port via the PORT env var)
subprocess.run([
    "streamlit", "run", "main.py",
    "--server.port", os.environ.get("PORT", "8501"),
    "--server.address", "0.0.0.0"
])
