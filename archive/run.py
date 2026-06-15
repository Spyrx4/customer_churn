import uvicorn
import subprocess
import time
import sys
import os

def run_services():
    """Start Backend (FastAPI) and Frontend (Streamlit)"""
    print("Memulai Customer Churn Analytics...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Backend
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print("Backend berjalan: http://localhost:8000")

    time.sleep(2)

    # Frontend  
    print("Frontend memuat...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=project_root)
    except KeyboardInterrupt:
        print("\nBerhenti...")
    finally:
        backend_process.terminate()
        print("Selesai.")

if __name__ == "__main__":
    run_services()
