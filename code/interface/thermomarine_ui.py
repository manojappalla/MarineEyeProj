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
        self.boundaryShpPathTrainBtn.clicked.connect(self.selectShapefile)
        self.modelOutputPathTrainBtn.clicked.connect(self.selectModelFolder)
        # self.trainModelBtn.clicked.connect(self.trainModel)
        self.boundaryShpPathPlotBtn.clicked.connect(self.selectShapefile)
        self.modelPathPlotBtn.clicked.connect(self.selectModel)
        # self.plotGraphsBtn.clicked.connect(self.modelApplier)

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
        if fileName:
            self.boundaryShpPathPlotTxt.setText(fileName)

    def selectModelFolder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder", ""
        )
        if folder:
            self.output_folder = folder
            self.modelOutputPathTrainTxt.setText(folder)

    def selectModel(self):
        """Open a file dialog to select a Model file (.pkl)"""
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Model",
            "",
            "Shapefiles (*.pkl)",
            options=options,
        )
        if fileName:
            sender = self.sender()
            if sender == self.modelPathPlotBtn:
                self.modelPathPlotTxt.setText(fileName)