from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import  QtGui, uic
import marineeye.floodcast as fpm
import os
import pandas as pd
import numpy as np


class FloodcastWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load UI dynamically
        uic.loadUi("ui/floodcast.ui", self)

        self.signal_window_initialization()

    def signal_window_initialization(self):
        self.centralWidget().layout().setContentsMargins(10, 10, 10, 10)
        self.Progress_Bar.setRange(0,6)
        self.image_path = os.path.join("icons", "MarineEye_Background.png")
        self.label.setPixmap(QtGui.QPixmap(self.image_path))
        self.Execute_Code.clicked.connect(self.application_pipeline)

         
    def collect_window_data(self):
        self.start_date = pd.to_datetime(self.Date_Input.date().toPyDate())
        self.minimum_longitude = float(self.Minimum_Longitude_Input.text())
        self.maximum_longitude = float(self.Maximum_Longitude_Input.text())
        self.minimum_latitude = float(self.Minimum_Latitude_Input.text())
        self.maximum_latitude = float(self.Maximum_Latitude_Input.text())
        self.alt_weight_wrt_wind_stress = int(self.SSH_Wind_Stress_Weight_Input.value())
        self.alt_weight_wrt_air_density = int(self.SSH_Air_Density_Wight_Input.value())
        self.wind_stress_weight_wrt_air_density = int(self.Wind_Stress_Air_Density_Weight_Input.value())

    def append_label(self, new_line,k):
        if k==1:
            current_text = self.User_Display_Screen.text()
            self.User_Display_Screen.setText(current_text + "<br>" + new_line)
        else:
            self.User_Display_Screen.setText(new_line)

    def application_pipeline(self):
        comment = "Process Initiated..."
        self.append_label(comment,0)
        self.Progress_Bar.setValue(0)
        self.collect_window_data()
        self.Progress_Bar.setValue(1)

        final_path = os.path.join("data", "floodcast", "Monthly_Averages", "Meta_Data_Monthly_Avg.csv")
        self.current_monthly_avg_year_data = np.loadtxt(final_path)
        user_input_data = [self.start_date.year,self.minimum_longitude,self.maximum_longitude,self.minimum_latitude,self.maximum_latitude]

        if np.array_equal(user_input_data,self.current_monthly_avg_year_data):
            comment = (f"Monthly average for the year {self.start_date.year} has already been calculated and been stored!")
            self.append_label(comment,1)
        else:
            comment = (f"Monthly average for the year {self.start_date.year} has not been calculated hence, starting the process to calculate!")
            self.append_label(comment,1)
            fpm.calc_monthly_mean(self.start_date.year,self.minimum_longitude,self.maximum_longitude,self.minimum_latitude,self.maximum_latitude)
            comment = ("Monthly averages has been calculated and saved for all parameters!")
            self.append_label(comment,1)
            np.savetxt(final_path, user_input_data, fmt='%.10f')

        self.Progress_Bar.setValue(2)
        self.end_date = self.start_date; 
        [self.alt_data, self.wind_stress_data, self.air_density_data]=fpm.collect_data(self.start_date,self.end_date,self.minimum_longitude,self.maximum_longitude,self.minimum_latitude,self.maximum_latitude)
        comment = "Data Collection for the Entered Date is Successfull"
        self.append_label(comment,1)
        self.Progress_Bar.setValue(3)
        [processed_matrix,anomaly_matrix,monthly_mean_AHP,CR_comment] = fpm.AHP_and_Anomaly_process(self.start_date.month,self.alt_data, self.wind_stress_data, self.air_density_data,self.alt_weight_wrt_wind_stress,self.alt_weight_wrt_air_density,self.wind_stress_weight_wrt_air_density)
        self.append_label(CR_comment,1)
        comment = "AHP Process Completed and Weights Obtained Successfully"
        self.append_label(comment,1)
        self.Progress_Bar.setValue(4)
        fpm.plot_heat_map(processed_matrix,anomaly_matrix,monthly_mean_AHP)
        self.Progress_Bar.setValue(5)
        self.image_path = os.path.join("icons", "Flood_Anomaly_Map.png")
        self.label.setPixmap(QtGui.QPixmap(self.image_path))
        self.Progress_Bar.setValue(6)
        comment = "Flood Index Maps are Plotted & Run is Successfull"
        self.append_label(comment,1)
        threshold = 0.53
        val = anomaly_matrix[:,-1].flatten()
        val[val<threshold] = 0
        if np.all(val == 0):
            comment = (f"Verdict: No Flood Possibility is Identified for the date {self.start_date}")
        else:
            comment = (f"Verdict: Flood Possibility is Identified for the date {self.start_date}")
        self.append_label(comment,1)
      

# if __name__ == "__main__":
#     app = QApplication([])
#     window = FloodcastWindow()
#     window.show()
#     app.exec()