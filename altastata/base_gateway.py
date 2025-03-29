import os
import subprocess
import time
import socket
import pkg_resources
from py4j.java_gateway import JavaGateway, GatewayParameters
from threading import Thread
import platform

class BaseGateway:
    def __init__(self, port=25333):
        self.port = port
        self.started_server = False  # Track if the server was started by this instance
        if not self._is_gateway_running():
            self._start_gateway_server()
            self.started_server = True  # Indicate that this instance started the server

        # Attempt to connect to the JVM with retries
        for _ in range(10):  # Try 10 times
            try:
                self.gateway = JavaGateway(gateway_parameters=GatewayParameters(port=self.port, auto_convert=True))
                break
            except Exception as e:
                print(f"Connection failed: {e}, retrying...")
                time.sleep(2)  # Wait before retrying
        else:
            raise RuntimeError("Failed to connect to the Java GatewayServer after multiple attempts")

    def _is_gateway_running(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('127.0.0.1', self.port))
            return result == 0

    def _start_gateway_server(self):
        # Find the JAR directory using pkg_resources
        jar_dir = pkg_resources.resource_filename('altastata', 'lib')

        # List all JAR files in the directory
        jar_files = [os.path.join(jar_dir, f) for f in os.listdir(jar_dir) if f.endswith('.jar')]

        # Determine classpath format based on the OS
        if platform.system() == "Windows":
            classpath = ";".join(jar_files)  # Use semicolon for Windows
        else:
            classpath = ":".join(jar_files)  # Use colon for UNIX-like systems

        # Command to run the Java gateway server
        java_command = [
            'java',
            '--add-opens', 'java.base/java.util=ALL-UNNAMED',
            '-cp', classpath,
            'py4j.GatewayServer'
        ]
        print(f"Running command: {' '.join(java_command)}")  # Add this line

        # Start the Java gateway server
        self.process = subprocess.Popen(java_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Function to read the process output asynchronously
        def read_output(pipe):
            with pipe:
                for line in iter(pipe.readline, b''):
                    print(line.decode('utf-8').strip())

        # Start threads to read stdout and stderr
        Thread(target=read_output, args=(self.process.stdout,), daemon=True).start()
        Thread(target=read_output, args=(self.process.stderr,), daemon=True).start()

        # Give some time for the Java gateway to start
        time.sleep(2)

    def shutdown(self):
        self.gateway.shutdown()
        if self.started_server:
            if hasattr(self, 'process'):
                self.process.terminate()
