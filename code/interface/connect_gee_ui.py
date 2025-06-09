from PyQt5 import QtWidgets, uic
import sys
import ee
import io
import subprocess
import platform
from PyQt5.QtCore import QThread, pyqtSignal

PROJECT_NAME = None

# Worker thread for Earth Engine authentication
class EEAthenticationThread(QThread):
    output_signal = pyqtSignal(str)  # Signal to send output text

    def run(self):
        """Runs Earth Engine authentication and captures output dynamically."""
        self.output_signal.emit("Starting Earth Engine authentication...\n")

        try:
            if platform.system() == "Windows":
                # Windows: Use `subprocess` without pexpect
                process = subprocess.Popen(
                    [
                        ".venv/bin/python",
                        "-c",
                        "import ee; ee.Authenticate(auth_mode='localhost')",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    text=True,
                )

                # Read output line by line and forward to UI
                for line in iter(process.stdout.readline, ""):
                    self.output_signal.emit(line)

                for line in iter(process.stderr.readline, ""):
                    self.output_signal.emit(f"Error: {line}")

                process.wait()  # Wait for user to authenticate

            else:
                # Linux/macOS: Use `pexpect` to handle interactive prompts
                import pexpect

                child = pexpect.spawn(
                    ".venv/bin/python -c 'import ee; ee.Authenticate(auth_mode=\"localhost\")'",
                    encoding="utf-8",
                    timeout=600,
                )

                while True:
                    try:
                        line = child.readline()
                        if not line:
                            break
                        self.output_signal.emit(line)
                    except pexpect.EOF:
                        break

                child.wait()
            self.output_signal.emit("\nAuthentication process completed.\n")

        except Exception as e:
            self.output_signal.emit(f"Exception: {e}\n")
            self.output_signal.emit("\nAuthentication process Failed.\n")

        
        try:
            global PROJECT_NAME
            ee.Initialize(project=PROJECT_NAME)
            self.output_signal.emit("Initialization process completed.\n")
        except Exception as e:
            print(PROJECT_NAME)
            self.output_signal.emit(f"\n{e}: Error in Initialization.\n")


class ConnectGEEDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        # Load UI dynamically
        uic.loadUi("ui/connect_gee.ui", self)

        # Set up signals
        self.setupSignals()

        # Initialize authentication thread
        self.auth_thread = None

    def setupSignals(self):
        """Connect UI elements to their functions."""
        self.connectBtn.clicked.connect(self.connectGEE)

    def connectGEE(self):
        """Authenticate with Google Earth Engine and update the output text box."""

        # Disable button to prevent multiple clicks
        self.connectBtn.setEnabled(False)
        global PROJECT_NAME
        PROJECT_NAME = self.projectNameTxt.text()
        # Initialize authentication thread
        self.auth_thread = EEAthenticationThread()
        self.auth_thread.output_signal.connect(self.appendOutput)  # Connect signal
        self.auth_thread.start()  # Start the thread (calls `run()` internally)

    def appendOutput(self, text):
        """Append new output to the QTextEdit widget instead of replacing it."""
        self.connectConsoleOutput.appendPlainText(text)  # Append new text
        self.connectConsoleOutput.ensureCursorVisible()  # Auto-scroll to bottom

        # Re-enable the authenticate button after completion
        self.connectBtn.setEnabled(True)