from PyQt5 import QtWidgets, uic
import sys
import ee
import io
import subprocess
import platform
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication
import plotly.graph_objects as go
from marineeye.thermomarine import Config, DataHelper, ModelHelper, Visualizer
import geopandas as gpd
from pathlib import Path
import joblib


class ThermomarineDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        # Load UI dynamically
        uic.loadUi("ui/thermomarine.ui", self)

        # Set up signals
        self.setupSignals()

    def setupSignals(self):
        self.boundaryShpPathTrainBtn.clicked.connect(self.selectShapefileTrain)
        self.modelOutputPathTrainBtn.clicked.connect(self.selectModelFolder)
        self.scalarOutputPathTrainBtn.clicked.connect(self.selectScalarFolder)
        self.trainModelBtn.clicked.connect(self.trainModel)
        self.boundaryShpPathPlotBtn.clicked.connect(self.selectShapefilePlot)
        self.modelPathPlotBtn.clicked.connect(self.selectModel)
        self.scalarPathBtn.clicked.connect(self.selectScalar)
        # self.plotGraphsBtn.clicked.connect(self.modelApplier)

    def selectShapefileTrain(self):
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
            self.boundaryShpPathTrainTxt.setText(fileName)

    def selectShapefilePlot(self):
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

    def selectScalarFolder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder", ""
        )
        if folder:
            self.output_folder = folder
            self.scalarOutputPathTrainTxt.setText(folder)

    def selectModel(self):
        """Open a file dialog to select a Model file (.keras)"""
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Model",
            "",
            "Shapefiles (*.keras)",
            options=options,
        )
        if fileName:
            sender = self.sender()
            if sender == self.modelPathPlotBtn:
                self.modelPathPlotTxt.setText(fileName)

    def selectScalar(self):
        """Open a file dialog to select a Model file (.pkl)"""
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Scalar",
            "",
            "Shapefiles (*.pkl)",
            options=options,
        )
        if fileName:
            sender = self.sender()
            if sender == self.modelPathPlotBtn:
                self.modelPathPlotTxt.setText(fileName)

    def trainModel(self):
        START_DATE = self.startDateTrain.date().toString("yyyy-MM-dd")
        END_DATE = self.endDateTrain.date().toString("yyyy-MM-dd")

        shape_file_path = self.boundaryShpPathTrainTxt.text()
        # 1️⃣  Define bounding-box edges (plain floats, not tuples)
        gdf = gpd.read_file(shape_file_path)
        # 2. Extract bounds: [minx, miny, maxx, maxy]
        MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = gdf.total_bounds

        VAR_NAME = "analysed_sst"
        N_LAGS = self.nlagsSpinBox.value()
        EPOCHS = self.epochsSpinBox.value()

        # 2️⃣  Assemble the list the rest of the code expects
        BBOX = [MIN_LON, MAX_LON, MIN_LAT, MAX_LAT]

        # file names to save artefacts
        MODEL_BEST = str(Path(self.modelOutputPathTrainTxt.text()) / "best.keras")
        MODEL_FULL = str(Path(self.modelOutputPathTrainTxt.text()) / "thermomarine_final.keras")
        SCALER_PKL = str(Path(self.scalarOutputPathTrainTxt.text()) / "scaler.pkl")

        DataHelper.set_seed()

        # 1. Data
        ds = DataHelper.download_dataset(START_DATE, END_DATE)
        ts = DataHelper.load_ts(ds, VAR_NAME, BBOX, START_DATE, END_DATE)
        X, y, scaler = DataHelper.windowize(ts, N_LAGS)

        # 2. Model
        model = ModelHelper.build(N_LAGS)
        ModelHelper.train(model, X, y, epochs=EPOCHS)  # ← added missing )

        # 3. Save
        ModelHelper.save_final(model)  # thermomarine_final.keras
        joblib.dump(scaler, "scaler.pkl")  # save the fitted scaler
        Path(ModelHelper.CHECKPOINT).rename("best.keras")  # rename best checkpoint