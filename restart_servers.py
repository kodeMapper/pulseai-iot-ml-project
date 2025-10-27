import subprocess
import os
import platform

def restart_servers():
    """
    Restarts the Flask backend and React frontend servers by running them
    in new console windows. This version uses the CREATE_NEW_CONSOLE flag
    for better reliability on Windows.
    """
    print("--- Stopping existing server processes ---")

    # --- Stop Backend (Python/Flask) ---
    # Using taskkill to forcefully stop any running Python processes.
    print("Stopping Flask backend (python.exe)...")
    subprocess.run('taskkill /F /IM python.exe', shell=True, capture_output=True)
    
    # --- Stop Frontend (Node/React) ---
    # Using taskkill to forcefully stop any running Node.js processes.
    print("Stopping React frontend (node.exe)...")
    subprocess.run("taskkill /F /IM node.exe", shell=True, capture_output=True)

    print("\n--- Starting servers in new terminal windows ---")
    
    # Get absolute paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(project_root, '.venv', 'Scripts', 'python.exe')
    backend_script = os.path.join(project_root, 'webapp', 'backend', 'app.py')
    frontend_dir = os.path.join(project_root, 'webapp', 'frontend')

    # This flag is specific to Windows and tells subprocess to open a new window
    CREATE_NEW_CONSOLE = 0x00000010

    # --- Start Backend ---
    print("Starting Flask backend in a new window...")
    # We pass the command as a list of arguments.
    backend_command = [venv_python, backend_script]
    subprocess.Popen(backend_command, creationflags=CREATE_NEW_CONSOLE)

    # --- Start Frontend ---
    print("Starting React frontend in a new window...")
    # For npm, we need to run it from the correct directory (cwd) and use shell=True
    # because 'npm' is typically a .cmd file on Windows.
    frontend_command = "npm start"
    subprocess.Popen(frontend_command, cwd=frontend_dir, creationflags=CREATE_NEW_CONSOLE, shell=True)

    print("\n--- Servers are restarting in new console windows ---")
    print("It may take a moment for them to become fully active.")

if __name__ == '__main__':
    if platform.system() != "Windows":
        print("This script is designed for Windows and may not work on other operating systems.")
    else:
        restart_servers()
