from typing import List, Tuple, Dict, Optional
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QColorDialog,
    QHeaderView,
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
import random
import numpy as np


# ------------------------------------------
# Symbology Dialog for Continuous Data
# ------------------------------------------
class SymbologyDialogContinuous(QDialog):
    """
    Dialog for defining class breaks and selecting colors for continuous raster data.
    """

    def __init__(self, min_val: float, max_val: float, num_classes: int = 4):
        """
        Initializes the continuous symbology dialog.

        Parameters:
        - min_val (float): Minimum value of the raster.
        - max_val (float): Maximum value of the raster.
        - num_classes (int): Initial number of classes to display.
        """
        super().__init__()
        self.setWindowTitle("Symbology - Continuous Data")
        self.resize(400, 300)

        self.min_val = min_val
        self.max_val = max_val
        self.num_classes = num_classes
        self.ranges = []
        self.color_map = []

        layout = QVBoxLayout()

        # Class count spinner
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Number of Classes:"))
        self.class_spinner = QSpinBox()
        self.class_spinner.setMinimum(2)
        self.class_spinner.setMaximum(10)
        self.class_spinner.setValue(num_classes)
        self.class_spinner.valueChanged.connect(self.build_table)
        top_layout.addWidget(self.class_spinner)
        layout.addLayout(top_layout)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["From", "To", "Color"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        # Apply button
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.accept)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)
        self.build_table(num_classes)

    def build_table(self, n: int) -> None:
        """
        Builds the table rows with equally spaced ranges and default colors.

        Parameters:
        - n (int): Number of class intervals.
        """
        self.table.setRowCount(n)
        self.ranges = []
        self.color_map = []

        step = (self.max_val - self.min_val) / n
        for i in range(n):
            start = self.min_val + i * step
            end = start + step if i < n - 1 else self.max_val
            self.ranges.append((start, end))

            self.table.setItem(i, 0, QTableWidgetItem(f"{start:.2f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{end:.2f}"))

            color = QColor.fromHsv(int(360 * i / n), 255, 200)
            self.color_map.append(color)
            color_btn = QPushButton()
            color_btn.setStyleSheet(f"background-color: {color.name()}")
            color_btn.clicked.connect(lambda _, row=i: self.pick_color(row))
            self.table.setCellWidget(i, 2, color_btn)

    def pick_color(self, row: int) -> None:
        """
        Opens a color picker to change the color for a specific range.

        Parameters:
        - row (int): Table row to update the color for.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.color_map[row] = color
            btn = self.table.cellWidget(row, 2)
            btn.setStyleSheet(f"background-color: {color.name()}")

    def get_ranges_and_colors(self) -> Tuple[List[Tuple[float, float]], List[QColor]]:
        """
        Returns the selected value ranges and their corresponding colors.

        Returns:
        - tuple: (list of (from, to) tuples, list of QColor objects)
        """
        return self.ranges, self.color_map


# ------------------------------------------
# Symbology Dialog for Discrete (Classified) Data
# ------------------------------------------
class SymbologyDialogDiscrete(QDialog):
    """
    Dialog for selecting colors for classified (discrete) raster values.
    """

    def __init__(
        self,
        class_values: List[int],
        current_colors: Optional[Dict[int, QColor]] = None,
    ):
        """
        Initializes the discrete symbology dialog.

        Parameters:
        - class_values (list): List of unique class values.
        - current_colors (dict, optional): Mapping of class value to QColor.
        """
        super().__init__()
        self.setWindowTitle("Symbology - Classified Data")
        self.resize(400, 300)

        self.class_values = class_values
        self.color_map = {}

        layout = QVBoxLayout()
        self.table = QTableWidget(len(class_values), 2)
        self.table.setHorizontalHeaderLabels(["Class", "Color"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for i, val in enumerate(class_values):
            self.table.setItem(i, 0, QTableWidgetItem(str(val)))

            color = (
                current_colors[val]
                if current_colors and val in current_colors
                else QColor(*[random.randint(0, 255) for _ in range(3)])
            )
            self.color_map[val] = color

            btn = QPushButton()
            btn.setStyleSheet(f"background-color: {color.name()}")
            btn.clicked.connect(lambda _, v=val, b=btn: self.pick_color(v, b))
            self.table.setCellWidget(i, 1, btn)

        layout.addWidget(self.table)

        # Apply button
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.accept)
        layout.addWidget(apply_btn)
        self.setLayout(layout)

    def pick_color(self, class_val: int, button: QPushButton) -> None:
        """
        Opens a color picker for a specific class and updates the color.

        Parameters:
        - class_val (int or str): The class value to change color for.
        - button (QPushButton): The button to update with the new color.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.color_map[class_val] = color
            button.setStyleSheet(f"background-color: {color.name()}")

    def get_color_map(self) -> Dict[int, QColor]:
        """
        Returns the final color mapping after user selection.

        Returns:
        - dict: Mapping of class values to QColor.
        """
        return self.color_map
