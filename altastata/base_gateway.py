import os
import subprocess
import time
import socket
import pkg_resources
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters
from threading import Thread
import platform

class BaseGateway:
    def __init__(self, port=25333, enable_callback_server=True, callback_server_port=None):
        self.port = port
        self.enable_callback_server = enable_callback_server
        self.callback_server_port = callback_server_port
        self.started_server = False  # Track if the server was started by this instance
        if not self._is_gateway_running():
            self._start_gateway_server()
            self.started_server = True  # Indicate that this instance started the server

        # Attempt to connect to the JVM with retries
        for _ in range(10):  # Try 10 times
            try:
                if self.enable_callback_server:
                    # With callback server for event listeners
                    if self.callback_server_port:
                        # Use custom callback server port
                        self.gateway = JavaGateway(
                            gateway_parameters=GatewayParameters(port=self.port, auto_convert=True),
                            callback_server_parameters=CallbackServerParameters(
                                address="127.0.0.1",  # Java will connect to this address
                                port=self.callback_server_port,
                                daemonize=True,
                                daemonize_connections=True
                            )
                        )
                        # Explicitly start the callback server
                        self.gateway.start_callback_server()
                        print(f"✅ Callback server started on 127.0.0.1:{self.callback_server_port}")
                    else:
                        # Use default callback server port (0 = auto-select)
                        self.gateway = JavaGateway(
                            gateway_parameters=GatewayParameters(port=self.port, auto_convert=True),
                            callback_server_parameters=CallbackServerParameters(
                                address="127.0.0.1",
                                port=0,
                                daemonize=True,
                                daemonize_connections=True
                            )
                        )
                        # Explicitly start the callback server
                        self.gateway.start_callback_server()
                        actual_port = self.gateway.get_callback_server().get_listening_port()
                        print(f"✅ Callback server started on 127.0.0.1:{actual_port}")
                else:
                    # Without callback server (for clients that don't listen to events)
                    self.gateway = JavaGateway(
                        gateway_parameters=GatewayParameters(port=self.port, auto_convert=True)
                    )
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
            '-Xms1g',                    # Initial heap size
            '-Xmx4g',                    # Max heap size
            #'-XX:+UseZGC',               # Use ZGC for very low pause times - !!! crashes on IBM Z and LinuxONE !!!
            '-XX:+UnlockExperimentalVMOptions',
            '-XX:+UseStringDeduplication', # Reduce memory usage for strings
            '-Xlog:gc*:file=gc.log:time,uptime:filecount=5,filesize=10M', # GC logging
            '-XX:ThreadStackSize=256k',  # Reduce thread stack size
            '-XX:+DisableExplicitGC',    # Prevent explicit GC calls
            '-cp', classpath,
            'py4j.GatewayServer',
            str(self.port)              # Tell Java which port to listen on
        ]
        
        print(f"Running command: {' '.join(java_command)}")

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
