import sys
import os

# Equivalent to setting PYTHONPATH=/app/code
code_path = "code"
if code_path not in sys.path:
    sys.path.insert(0, code_path)

from PyQt5.QtCore import QCoreApplication, Qt

QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
import PyQt5.QtWebEngineWidgets

import sys
import rasterio
import numpy as np
from pyproj import Transformer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QTreeWidgetItem,
    QMenu,
    QAction,
    QDialog,
    QVBoxLayout,
    QTextEdit,
    QGraphicsView,
    QLineEdit,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic
from marineeye.symbology import SymbologyDialogDiscrete, SymbologyDialogContinuous

# import ui classes
from connect_gee_ui import ConnectGEEDialog
from mangrover_ui import MangroverDialog
from floodcast_ui import FloodcastWindow
from connect_cmems_ui import ConnectCMEMSDialog
from algal_bloom_ui import AlgalDashboard


class HoverGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.main_window = parent

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        x, y = pos.x(), pos.y()
        if self.main_window:
            self.main_window.update_mouse_position(x, y)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/main.ui", self)

        self.scene = QGraphicsScene()
        self.layer_items = {}
        self.raster_crs = None
        self.raster_transform = None
        self.transformer = None

        self.setupGraphicsView()
        self.setupSignals()
        self.setupLayerTreeContextMenu()

    def setupGraphicsView(self):
        orig_view = self.mapView
        self.mapView = HoverGraphicsView(self)
        self.mapView.setObjectName("mapView")
        self.mapView.setScene(self.scene)

        layout = orig_view.parent().layout()
        layout.replaceWidget(orig_view, self.mapView)
        orig_view.deleteLater()

    def setupSignals(self):
        self.actionOpen.triggered.connect(self.open_raster)
        self.layerTree.itemChanged.connect(self.handle_layer_visibility)

        # Signals to open Modules and UIs
        self.actionGEE.triggered.connect(self.openConnectGEE)
        self.actionMangrover.triggered.connect(self.openMangrover)
        self.actionFloodcast.triggered.connect(self.openFloodcast)
        self.actionCMEMS.triggered.connect(self.openConnectCMEMS)
        self.actionAlgalBloom.triggered.connect(self.openAlgalBloom)

    # *************************** Execute Modules Here ********************************
    def openConnectGEE(self):
        # # Create an instance of the SymbologyDialogDiscrete and show it
        self.connect_gee_dialog = ConnectGEEDialog()
        self.connect_gee_dialog.exec_()  # Using exec_() for modal dialog

    def openMangrover(self):
        # # Create an instance of the SymbologyDialogDiscrete and show it
        self.MangroveDialog = MangroverDialog()
        self.MangroveDialog.exec_()  # Using exec_() for modal dialog

    def openFloodcast(self):
        self.FloodcastWindow = FloodcastWindow()
        self.FloodcastWindow.show()  

    def openConnectCMEMS(self):
        self.connect_cmems_dialog = ConnectCMEMSDialog()
        self.connect_cmems_dialog.exec_()

    def openAlgalBloom(self):
        # # Create an instance of the SymbologyDialogDiscrete and show it
        self.AlgalDashboard = AlgalDashboard()
        self.AlgalDashboard.show()  # Using exec_() for modal dialog

    # *********************************************************************************

    def setupLayerTreeContextMenu(self):
        self.layerTree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.layerTree.customContextMenuRequested.connect(self.show_layer_context_menu)

    def open_raster(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Raster", "", "Raster files (*.tif *.tiff)"
        )
        if not file_path or file_path in self.layer_items:
            return

        with rasterio.open(file_path) as src:
            self.raster_crs = src.crs
            self.raster_transform = src.transform
            self.transformer = Transformer.from_crs(
                self.raster_crs, "EPSG:4326", always_xy=True
            )
            count = src.count
            if count >= 3:
                r = src.read(1)
                g = src.read(2)
                b = src.read(3)
                rgb = np.stack((r, g, b), axis=-1)
                img = self._normalize(rgb)
            elif count == 1:
                band = src.read(1)
                norm = self._normalize(band)
                img = np.stack((norm, norm, norm), axis=-1)
            else:
                return

        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(item)
        self.layer_items[file_path] = item

        layer_name = file_path.split("/")[-1]
        tree_item = self._make_tree_item(layer_name)
        self.layerTree.insertTopLevelItem(0, tree_item)
        self.update_z_values()
        epsg_code = self.raster_crs.to_epsg()
        # self.crsTxt.setText(f"EPSG:{epsg_code}")

    def _normalize(self, arr):
        arr = arr.astype(float)
        min_val, max_val = np.nanmin(arr), np.nanmax(arr)
        norm = ((arr - min_val) / (max_val - min_val + 1e-6)) * 255
        return norm.astype(np.uint8)

    def _make_tree_item(self, name):
        item = QTreeWidgetItem()
        item.setText(0, name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Checked)
        return item

    def show_layer_context_menu(self, pos):
        item = self.layerTree.itemAt(pos)
        if not item:
            return

        layer_name = item.text(0)
        menu = QMenu()
        zoom_action = QAction("Zoom to Layer", self)
        remove_action = QAction("Remove Layer", self)
        info_action = QAction("Information", self)
        symbology_action = QAction("Symbology", self)

        menu.addAction(zoom_action)
        menu.addAction(remove_action)
        menu.addAction(info_action)
        menu.addAction(symbology_action)

        action = menu.exec_(self.layerTree.viewport().mapToGlobal(pos))

        if action == remove_action:
            self.remove_layer(layer_name)
        elif action == zoom_action:
            self.zoom_to_layer(layer_name)
        elif action == info_action:
            self.show_layer_info(layer_name)
        elif action == symbology_action:
            self.open_symbology(layer_name)

    def open_symbology(self, layer_name):
        for path, item in self.layer_items.items():
            if path.endswith(layer_name):
                with rasterio.open(path) as src:
                    band = src.read(1)
                    unique_vals = np.unique(band)

                    is_discrete = (
                        band.dtype in [np.uint8, np.int16, np.int32]
                        and len(unique_vals) <= 20
                    )

                    if is_discrete:
                        # Discrete symbology
                        dialog = SymbologyDialogDiscrete(list(unique_vals))
                        if dialog.exec_():
                            color_map = dialog.get_color_map()
                            colored = np.zeros((*band.shape, 3), dtype=np.uint8)
                            for val, color in color_map.items():
                                mask = band == val
                                colored[mask] = [
                                    color.red(),
                                    color.green(),
                                    color.blue(),
                                ]
                    else:
                        # Continuous symbology
                        min_val, max_val = np.nanmin(band), np.nanmax(band)
                        dialog = SymbologyDialogContinuous(
                            min_val, max_val, num_classes=4
                        )
                        if dialog.exec_():
                            ranges, colors = dialog.get_ranges_and_colors()
                            colored = np.zeros((*band.shape, 3), dtype=np.uint8)
                            for (low, high), color in zip(ranges, colors):
                                mask = (band >= low) & (band <= high)
                                colored[mask] = [
                                    color.red(),
                                    color.green(),
                                    color.blue(),
                                ]

                    # Apply new image
                    h, w, _ = colored.shape
                    qimg = QImage(colored.data, w, h, 3 * w, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    self.layer_items[path].setPixmap(pixmap)
                break

    def remove_layer(self, layer_name):
        path_to_remove = None
        for path in self.layer_items:
            if path.endswith(layer_name):
                path_to_remove = path
                break

        if path_to_remove:
            item = self.layer_items.pop(path_to_remove)
            self.scene.removeItem(item)

            for i in range(self.layerTree.topLevelItemCount()):
                tree_item = self.layerTree.topLevelItem(i)
                if tree_item.text(0) == layer_name:
                    self.layerTree.takeTopLevelItem(i)
                    break

        self.update_z_values()

    def zoom_to_layer(self, layer_name):
        for path, item in self.layer_items.items():
            if path.endswith(layer_name):
                bounds = item.boundingRect()
                margin_x = bounds.width() * 0.1
                margin_y = bounds.height() * 0.1
                expanded = bounds.adjusted(-margin_x, -margin_y, margin_x, margin_y)
                self.mapView.fitInView(expanded, Qt.KeepAspectRatio)
                break

    def handle_layer_visibility(self, item, column):
        layer_name = item.text(0)
        is_checked = item.checkState(0) == Qt.Checked

        for path, pixmap_item in self.layer_items.items():
            if path.endswith(layer_name):
                pixmap_item.setVisible(is_checked)
                break

    def update_z_values(self):
        total = self.layerTree.topLevelItemCount()
        for i in range(total):
            item = self.layerTree.topLevelItem(i)
            layer_name = item.text(0)
            for path, graphic_item in self.layer_items.items():
                if path.endswith(layer_name):
                    graphic_item.setZValue(total - i)
                    break

    def show_layer_info(self, layer_name):
        for path in self.layer_items:
            if path.endswith(layer_name):
                try:
                    with rasterio.open(path) as src:
                        meta = src.meta
                        metadata_str = "\n".join(f"{k}: {v}" for k, v in meta.items())
                        self.show_metadata_dialog(layer_name, metadata_str)
                except Exception as e:
                    self.show_metadata_dialog(
                        layer_name, f"Error reading metadata:\n{e}"
                    )
                break

    def show_metadata_dialog(self, title, message):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Metadata - {title}")
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(message)
        layout.addWidget(text_edit)
        dialog.setLayout(layout)
        dialog.resize(500, 400)
        dialog.exec_()

    def update_mouse_position(self, x, y):
        if self.raster_transform and self.transformer:
            try:
                col, row = int(x), int(y)
                map_x, map_y = self.raster_transform * (col, row)
                lon, lat = self.transformer.transform(map_x, map_y)
                self.coordinatesTxt.setText(f"Lat: {lat:.3f}, Lon: {lon:.3f}")
            except Exception:
                self.coordinatesTxt.setText("Invalid")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
