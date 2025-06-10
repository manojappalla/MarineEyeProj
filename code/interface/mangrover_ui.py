from PyQt5 import QtWidgets, uic
import sys
import ee
import io
import subprocess
import platform
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication
import plotly.graph_objects as go
from marineeye.mangrover import *


class MangroverDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        # Load UI dynamically
        uic.loadUi("ui/mangrover.ui", self)

        # Set up signals
        self.setupSignals()

    def setupSignals(self):
        self.boundaryShpPathTrainBtn.clicked.connect(self.selectShapefile)
        self.trainShpPathBtn.clicked.connect(self.selectShapefile)
        self.modelOutputPathTrainBtn.clicked.connect(self.selectModelFolder)
        self.trainModelBtn.clicked.connect(self.trainModel)
        self.boundaryShpPathPlotBtn.clicked.connect(self.selectShapefile)
        self.modelPathPlotBtn.clicked.connect(self.selectModel)
        self.plotGraphsBtn.clicked.connect(self.modelApplier)

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
            sender = self.sender()
            if sender == self.boundaryShpPathTrainBtn:
                self.boundaryShpPathTrainTxt.setText(fileName)
            elif sender == self.trainShpPathBtn:
                self.trainShpPathTxt.setText(fileName)
                # Load shapefile and extract attribute names
                gdf = gpd.read_file(fileName)
                attribute_names = [col for col in gdf.columns if col != "geometry"]
                # Populate combo box
                self.classVarComboBox.clear()
                self.classVarComboBox.addItems(attribute_names)
            elif sender == self.boundaryShpPathPlotBtn:
                self.boundaryShpPathPlotTxt.setText(fileName)

    def selectModelFolder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder", ""
        )
        if folder:
            self.output_folder = folder
            self.modelOutputPathTrainTxt.setText(folder)

    def trainModel(self):
        self.progressBarTrain.setValue(1)
        config = Config(
            aoi_shapefile_path=self.boundaryShpPathTrainTxt.text(),
            training_shapefile_path=self.trainShpPathTxt.text(),
            start_date=self.startDateTrain.date().toString("yyyy-MM-dd"),
            end_date=self.endDateTrain.date().toString("yyyy-MM-dd"),
            bands=["SR_B5", "SR_B6", "SR_B4", "NDVI", "MNDWI", "SR", "GCVI", "SRTM"],
            model_output_path=self.modelOutputPathTrainTxt.text(),
        )
        processor = DataProcessor(config)
        image = processor.get_processed_image()
        self.progressBarTrain.setValue(25)
        sampler = SampleManager(image, config, self.classVarComboBox.currentText())
        self.progressBarTrain.setValue(50)
        train, test = sampler.sample_and_split(split=0.7)
        self.progressBarTrain.setValue(75)
        trainer = ModelTrainer(
            config.bands,
            config.model_output_path,
            class_attribute=self.classVarComboBox.currentText(),
        )
        classifier = trainer.train(train)
        trainer.save_metadata(classifier=classifier)
        self.progressBarTrain.setValue(100)

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

    def plot_mangrove_ndvi_time_series(self, df: pd.DataFrame) -> str:
        dates = pd.to_datetime(df["date"])
        ndvi_values = df["mean_ndvi"].astype(float)

        fig = go.Figure(
            data=go.Scatter(
                x=dates,
                y=ndvi_values,
                mode="lines+markers",
                marker=dict(color="darkgreen"),
                line=dict(width=2),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>NDVI: %{y:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Mangrove NDVI Over Time",
            xaxis_title="Date",
            yaxis_title="Mean NDVI",
            hovermode="x unified",
            template="plotly_white",
        )

        return fig.to_html(include_plotlyjs="cdn")

    def plot_mangrove_area_time_series(self, df: pd.DataFrame) -> str:
        dates = pd.to_datetime(df["date"])
        area_values = df["area_km2"].astype(float)

        fig = go.Figure(
            data=go.Scatter(
                x=dates,
                y=area_values,
                mode="lines+markers",
                marker=dict(color="seagreen"),
                line=dict(width=2),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Area: %{y:.2f} km²<extra></extra>",
            )
        )

        fig.update_layout(
            title="Mangrove Area Over Time",
            xaxis_title="Date",
            yaxis_title="Area (km²)",
            hovermode="x unified",
            template="plotly_white",
        )

        return fig.to_html(include_plotlyjs="cdn")

    def modelApplier(self):
        self.progressBarPlot.setValue(1)
        applier = ModelApplier(
            model_path=self.modelPathPlotTxt.text(),
            aoi_path=self.boundaryShpPathPlotTxt.text(),
            start_date=self.startDatePlot.date().toString("yyyy-MM-dd"),
            end_date=self.endDatePlot.date().toString("yyyy-MM-dd"),
            cloud_cover=self.cloudCoverPlotTxt.value(),
            bands=["SR_B5", "SR_B6", "SR_B4", "NDVI", "MNDWI", "SR", "GCVI", "SRTM"],
        )
        self.progressBarPlot.setValue(25)

        df = applier.run_parallel()
        self.progressBarPlot.setValue(75)
        print(df)
        html_area = self.plot_mangrove_area_time_series(df)
        html_ndvi = self.plot_mangrove_ndvi_time_series(df)
        self.areaPlot.setHtml(html_area)  # QWebEngineView for area
        self.ndviPlot.setHtml(html_ndvi)
        self.progressBarPlot.setValue(100)
