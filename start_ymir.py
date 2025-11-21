#!/usr/bin/env python3
"""
Y.M.I.R System Launcher
Starts all microservices in correct order with proper environment activation
"""

import subprocess
import time
import os
import sys
import signal
import platform
from pathlib import Path

class YMIRLauncher:
    def __init__(self):
        self.processes = []
        self.is_windows = platform.system() == "Windows"
        self.venv_path = Path(".venv")
        
        # Define the startup sequence
        self.services = [
            {
                "name": "Text Microservice",
                "script": "text_microservice_working.py",
                "port": 5003,
                "delay": 3
            },
            {
                "name": "Face Microservice", 
                "script": "face_microservice_working.py",
                "port": 5002,
                "delay": 5
            },
            {
                "name": "Main Application",
                "script": "app.py", 
                "port": 5000,
                "delay": 2
            }
        ]

    def get_python_command(self):
        """Get the correct Python command for the virtual environment"""
        if self.is_windows:
            python_cmd = self.venv_path / "Scripts" / "python.exe"
        else:
            python_cmd = self.venv_path / "bin" / "python"
        
        if python_cmd.exists():
            return str(python_cmd)
        else:
            print(f"⚠️  Virtual environment not found at {self.venv_path}")
            print("Using system Python...")
            return "python" if self.is_windows else "python3"

    def start_service(self, service):
        """Start a single microservice"""
        python_cmd = self.get_python_command()
        
        print(f"🚀 Starting {service['name']}...")
        print(f"   Command: {python_cmd} {service['script']}")
        print(f"   Port: {service['port']}")
        
        try:
            # Start the process
            if self.is_windows:
                process = subprocess.Popen(
                    [python_cmd, service['script']],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    cwd=os.getcwd()
                )
            else:
                process = subprocess.Popen(
                    [python_cmd, service['script']],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=os.getcwd()
                )
            
            self.processes.append({
                'process': process,
                'name': service['name'],
                'script': service['script']
            })
            
            print(f"✅ {service['name']} started (PID: {process.pid})")
            
            # Wait for service to initialize
            print(f"   Waiting {service['delay']} seconds for initialization...")
            time.sleep(service['delay'])
            
        except Exception as e:
            print(f"❌ Failed to start {service['name']}: {e}")
            return False
        
        return True

    def check_service_health(self, port):
        """Quick health check for a service"""
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def start_all_services(self):
        """Start all services in sequence"""
        print("=" * 60)
        print("🧠 Y.M.I.R AI Emotion Detection System Launcher")
        print("=" * 60)
        print()
        
        # Check if virtual environment exists
        if self.venv_path.exists():
            print(f"✅ Virtual environment found: {self.venv_path}")
        else:
            print(f"⚠️  Virtual environment not found at {self.venv_path}")
            print("   Using system Python...")
        print()
        
        # Start each service
        for i, service in enumerate(self.services, 1):
            print(f"[{i}/{len(self.services)}] {'-' * 40}")
            success = self.start_service(service)
            
            if not success:
                print(f"❌ Failed to start {service['name']}. Stopping...")
                self.cleanup()
                return False
            
            print()
        
        print("🎉 All services started successfully!")
        print()
        print("📍 Access URLs:")
        print("   • Main App: http://localhost:5000")
        print("   • AI Dashboard: http://localhost:5000/ai_app")
        print("   • Therapist Finder: http://localhost:5000/therapist_finder")
        print("   • Face Service: http://localhost:5002")
        print("   • Text Service: http://localhost:5003")
        print()
        print("💡 Press Ctrl+C to stop all services")
        
        return True

    def cleanup(self):
        """Stop all running processes"""
        print("\n🛑 Stopping all services...")
        
        for proc_info in self.processes:
            try:
                process = proc_info['process']
                name = proc_info['name']
                
                if process.poll() is None:  # Process is still running
                    print(f"   Stopping {name}...")
                    
                    if self.is_windows:
                        process.terminate()
                    else:
                        process.send_signal(signal.SIGTERM)
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                        print(f"   ✅ {name} stopped")
                    except subprocess.TimeoutExpired:
                        print(f"   ⚠️ Force killing {name}...")
                        process.kill()
                        process.wait()
                        print(f"   ✅ {name} force stopped")
                        
            except Exception as e:
                print(f"   ❌ Error stopping {proc_info['name']}: {e}")
        
        print("🏁 All services stopped")

    def run(self):
        """Main execution function"""
        try:
            success = self.start_all_services()
            
            if success:
                # Keep the launcher running and monitor processes
                try:
                    while True:
                        time.sleep(1)
                        
                        # Check if any process has died
                        for proc_info in self.processes:
                            if proc_info['process'].poll() is not None:
                                print(f"⚠️ {proc_info['name']} has stopped unexpectedly")
                        
                except KeyboardInterrupt:
                    print("\n\n👋 Received shutdown signal")
                
        except Exception as e:
            print(f"❌ Launcher error: {e}")
        
        finally:
            self.cleanup()

if __name__ == "__main__":
    launcher = YMIRLauncher()
    launcher.run()