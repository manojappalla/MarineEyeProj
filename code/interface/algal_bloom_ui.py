from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys
import os
from PyQt5.QtCore import QThread, pyqtSignal, QDate
from PyQt5.QtGui import QMovie
from marineeye.algal_bloom_main import Worker

# Main Application
class AlgalDashboard(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # self.ui = Ui_AlgalDashboard()

        # Load UI dynamically
        uic.loadUi("ui/algal_bloom.ui", self)
    
        # Connect signals
        self.run_button.clicked.connect(self.run_detection)
        
        # Set default date to today
        self.start_date.setDate(QtCore.QDate.currentDate())

    def validate_coordinates(self):
        try:
            min_lon = float(self.min_lon.text())
            max_lon = float(self.max_lon.text())
            min_lat = float(self.min_lat.text())
            max_lat = float(self.max_lat.text())
            
            if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
                raise ValueError("Longitude must be between -180 and 180")
            if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
                raise ValueError("Latitude must be between -90 and 90")
            if min_lon >= max_lon:
                raise ValueError("Min longitude must be less than max longitude")
            if min_lat >= max_lat:
                raise ValueError("Min latitude must be less than max latitude")
                
            return min_lon, max_lon, min_lat, max_lat
        except ValueError as e:
            self.output_log.append(f"[ERROR] {str(e)}")
            return None

    def run_detection(self):
        # Validate coordinates
        coords = self.validate_coordinates()
        if coords is None:
            return

        min_lon, max_lon, min_lat, max_lat = coords

        # Validate at least one variable is selected
        selected_vars = []
        if self.chk_chl.isChecked(): selected_vars.append("chl")
        if self.chk_sst.isChecked(): selected_vars.append("sst")
        if self.chk_sss.isChecked(): selected_vars.append("sss")
        
        if not selected_vars:
            self.output_log.append("[ERROR] Select at least one variable")
            return

        # Set environment variables
        os.environ['MIN_LON'] = str(min_lon)
        os.environ['MAX_LON'] = str(max_lon)
        os.environ['MIN_LAT'] = str(min_lat)
        os.environ['MAX_LAT'] = str(max_lat)
        
        start_date = self.start_date.date().toString("yyyy-MM-dd")
        end_date = self.start_date.date().addDays(self.days_spin.value()).toString("yyyy-MM-dd")
        os.environ['START_DATE'] = start_date
        os.environ['END_DATE'] = end_date
        os.environ['VARIABLES'] = ",".join(selected_vars)

        self.output_log.append(f"[INFO] Starting analysis for area:")
        self.output_log.append(f"Longitude: {min_lon} to {max_lon}")
        self.output_log.append(f"Latitude: {min_lat} to {max_lat}")
        self.output_log.append(f"Date range: {start_date} to {end_date}")
        self.output_log.append(f"Variables: {', '.join(selected_vars)}")

        self.worker = Worker(self.days_spin.value())
        self.worker.output_signal.connect(self.output_log.append)
        self.worker.done_signal.connect(self.load_animation)
        self.worker.start()

    def load_animation(self):
        if os.path.exists("risk_animation.gif"):
            movie = QMovie("risk_animation.gif")
            self.gif_viewer.setMovie(movie)
            movie.start()
        else:
            self.output_log.append("[ERROR] Failed to generate animation")

# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     window = AlgalDashboard()
#     window.show()
#     sys.exit(app.exec_())