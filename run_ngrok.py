# ===================================
# NGROK DEPLOYMENT SCRIPT
# This script starts the Streamlit app and exposes it via an Ngrok tunnel.
# ===================================

import os
import sys
from pyngrok import ngrok
import subprocess
import time

# Define the path to the Streamlit app
# Assumes this script is in the project root
STREAMLIT_APP_PATH = os.path.join("ui", "app.py")
# UPDATED: Changed port to 8502 to avoid conflicts
PORT = 8502

def main():
    """
    Main function to start Streamlit and create an Ngrok tunnel.
    """
    streamlit_process = None
    public_url = None # Initialize public_url to None

    print("🚀 Starting Streamlit app in the background...")
    
    # Start the Streamlit app as a separate process
    try:
        # Using subprocess to run streamlit without blocking
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", STREAMLIT_APP_PATH,
            "--server.port", str(PORT),
            "--server.headless", "true" # Runs Streamlit without opening a browser window
        ])
        print(f"✅ Streamlit process started with PID: {streamlit_process.pid}")
        
        # Give Streamlit a moment to start up
        time.sleep(5)

    except Exception as e:
        print(f"❌ Failed to start Streamlit app: {e}")
        return

    print(f"🔗 Creating Ngrok tunnel for port {PORT}...")
    
    try:
        # Create a public URL tunnel to the Streamlit app
        public_url = ngrok.connect(PORT)
        print("="*50)
        print("🎉 YOUR APP IS LIVE! 🎉")
        print(f"🔗 Public URL: {public_url}")
        print("="*50)
        print("ℹ️ Keep this window open to keep your app live.")
        print("ℹ️ Press Ctrl+C to stop the app and close the tunnel.")
        
        # Keep the script running until interrupted
        while True:
            time.sleep(1)
            
    except Exception as e:
        print(f"❌ Failed to create Ngrok tunnel: {e}")
        
    finally:
        # Clean up processes on exit
        print("\n shutting down...")
        # Check if public_url was successfully created before disconnecting
        if public_url:
            ngrok.disconnect(public_url)
        # Check if streamlit_process was successfully created before terminating
        if streamlit_process:
            streamlit_process.terminate()
        print("✅ Tunnel and Streamlit app closed.")

if __name__ == "__main__":
    main()

