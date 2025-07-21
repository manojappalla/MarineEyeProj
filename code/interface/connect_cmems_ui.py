from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtSignal
import copernicusmarine
import time
import threading


class ConnectCMEMSDialog(QtWidgets.QDialog):
    login_result_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        uic.loadUi("ui/connect_cmems.ui", self)
        self.setupSignals()

    def setupSignals(self):
        self.cmemsConnectBtn.clicked.connect(self.CMEMS_login_run)

    def CMEMS_login_exec(self):
        self.login_chk_key = copernicusmarine.login(username=self.cmems_username,password=self.cmems_password,check_credentials_valid=True)

    def CMEMS_login_run(self):
        self.login_chk_key = False
        self.cmems_username = self.usernameCMEMSTxt.text()
        self.cmems_password = self.passwordCMEMSTxt.text()

        exec_thread = threading.Thread(target=self.CMEMS_login_exec)
        exec_thread.daemon = True # <--- This is the crucial line!
        exec_thread.start()

        time.sleep(2) # This will wait for 5 seconds, and if the login is done within 3 sec, then okay, or else that thread will exit in 5 sec.
        if self.login_chk_key != True:
            self.login_indicator_label.setText('Incorrect Password or Username')
        elif self.login_chk_key == True:
            self.login_indicator_label.setText('Login Successfull')





