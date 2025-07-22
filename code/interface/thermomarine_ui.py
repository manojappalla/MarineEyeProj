from PyQt5 import QtWidgets, uic
import sys
import ee
import io
import subprocess
import platform
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication
import plotly.graph_objects as go
from marineeye.thermomarine import *


class ThermomarineDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        # Load UI dynamically
        uic.loadUi("ui/thermomarine.ui", self)

        # Set up signals
        self.setupSignals()

    def setupSignals(self):
        self.boundaryShpPathTimeseriesBtn.clicked.connect(self.selectShapefile)
        self.timeseriesPathBtn.clicked.connect(self.timeSeriesFolderPath)
        self.extractTimeSeriesBtn.clicked.connect(self.extractTimeSeries)
        self.timeseriesPathPlotBtn.clicked.connect(self.loadTimeseriesCSV)
        self.forecastPlotGraphsBtn.clicked.connect(self.forecastSST)

    def selectShapefile(self):
        """Open a file dialog to select a Shapefile."""
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Shapefile",
            "",
            "Shapefiles (*.shp)",
            options=options,
        )
        self.boundaryShpPathTimeseriesTxt.setText(fileName)

    def timeSeriesFolderPath(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Timeseries Output Folder", ""
        )
        if folder:
            self.output_folder = folder
            self.timeseriesPathTxt.setText(folder)

    def extractTimeSeries(self):
        self.progressBarTrain.setValue(1)
        extractor = SSTExtractor(
            shapefile_path=self.boundaryShpPathTimeseriesTxt.text(),
            start_date=self.startDateTimeseries.date().toString("yyyy-MM-dd"),
            end_date=self.endDateTimeseries.date().toString("yyyy-MM-dd"),
        )
        self.progressBarTrain.setValue(25)
        extractor.extract_time_series()
        self.progressBarTrain.setValue(75)
        extractor.save_csv(f"{self.timeseriesPathTxt.text()}/sst_timeseries.csv")
        self.progressBarTrain.setValue(100)

    def loadTimeseriesCSV(self):
        """Open a file dialog to select a Time Series CSV."""
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select CSV",
            "",
            "Shapefiles (*.csv)",
            options=options,
        )
        self.timeseriesPathPlotTxt.setText(fileName)

    def forecastSST(self):
        self.progressBarPlot.setValue(1)
        forecaster = SSTForecaster(f"{self.timeseriesPathPlotTxt.text()}")
        forecaster.train_prophet()
        self.progressBarPlot.setValue(50)
        forecaster.forecast_future(days=self.forecastForPlotSpinBox.value())
        self.progressBarPlot.setValue(100)
        forecast_graph = forecaster.plot_forecast(forecast_days=self.forecastForPlotSpinBox.value())
        self.sstPlot.setHtml(forecast_graph)
