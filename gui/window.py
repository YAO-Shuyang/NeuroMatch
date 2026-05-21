import sys
import os
import pickle
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QLineEdit,
    QSpinBox,
    QLabel,
    QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, QModelIndex, QTimer
from PyQt6.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from neuromatch.variables.regneuron import RegisteredNeuron
from neuromatch.variables import AllToAllList
from neuromatch.visualize.loctime import LocTimeCurve
from neuromatch.visualize.maze_graph import NRG, S2F


class DataFrameViewer(QMainWindow):
    """Simple read-only viewer for displaying a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DataFrame Viewer")
        self.setGeometry(100, 100, 800, 500)
        self.df = df
        self.initUI()

    def initUI(self):
        self.tableWidget = QTableWidget()
        self.setCentralWidget(self.tableWidget)

        n_rows, n_cols = self.df.shape
        self.tableWidget.setRowCount(n_rows)
        self.tableWidget.setColumnCount(n_cols)
        self.tableWidget.setHorizontalHeaderLabels(self.df.columns.astype(str).tolist())

        for j in range(n_cols):
            self.tableWidget.setColumnWidth(j, 70)

        for i in range(n_rows):
            for j in range(n_cols):
                value = self.df.iloc[i, j]

                if pd.isna(value):
                    text = ""
                elif isinstance(value, (int, np.integer)):
                    text = str(value)
                elif isinstance(value, (float, np.floating)):
                    text = str(int(value)) if float(value).is_integer() else str(value)
                else:
                    text = str(value)

                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tableWidget.setItem(i, j, item)


class NeuroMatchGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroMatch GUI v1.0 beta")
        self.setGeometry(100, 100, 1200, 600)
        self.initUI()

    def initUI(self):
        left_layout = QVBoxLayout()
        right_layout = QHBoxLayout()

        self.opt_content = None
        self.ori_content = None
        self.sour_content = None

        self.df_titles = None
        self.df = None
        self.data = None

        self.plot_range = None
        self._row, self._col = None, None
        self.log_name = None
        self._p_thre = 0.8
        self._ori_n = None
        self._excel_dirname = None
        self.autoSaveTimer = None

        # ---------- Load buttons ----------
        loadPklButton = QPushButton("Load PKL File")
        loadPklButton.clicked.connect(self.loadPklFile)

        loadXlsxButton = QPushButton("Load Excel File")
        loadXlsxButton.clicked.connect(self.loadXlsxFile)

        loadDataButton = QPushButton("Load Data")
        loadDataButton.clicked.connect(self.loadData)

        loadLogDirButton = QPushButton("Select Directory")
        loadLogDirButton.clicked.connect(self.configureLogger)

        # ---------- Run / plot buttons ----------
        RunButton = QPushButton("Run")
        RunButton.clicked.connect(self.run)

        PlotButton = QPushButton("Plot")
        PlotButton.clicked.connect(self.plot)

        # ---------- Operation buttons ----------
        FillButton = QPushButton("Fill")
        FillButton.clicked.connect(self.fill)

        ReplaceButton = QPushButton("Replace")
        ReplaceButton.clicked.connect(self.replace)

        MoveOutButton = QPushButton("Move Out")
        MoveOutButton.clicked.connect(self.moveOut)

        AdoptButton = QPushButton("Adopt")
        AdoptButton.clicked.connect(self.adopt)

        ManualChangeButton = QPushButton("Change")
        ManualChangeButton.clicked.connect(self.manualChange)

        ChangeButtonLayout = QHBoxLayout()
        ChangeButtonLayout.addWidget(FillButton)
        ChangeButtonLayout.addWidget(ReplaceButton)
        ChangeButtonLayout.addWidget(MoveOutButton)
        ChangeButtonLayout.addWidget(AdoptButton)

        # ---------- Save / view buttons ----------
        ViewButton = QPushButton("View")
        ViewButton.clicked.connect(self.view)

        SaveButton = QPushButton("Save")
        SaveButton.clicked.connect(self.save)

        # ---------- Path line edits ----------
        self.pklFilePathLineEdit = QLineEdit()
        self.pklFilePathLineEdit.setReadOnly(True)

        self.xlsxFilePathLineEdit = QLineEdit()
        self.xlsxFilePathLineEdit.setReadOnly(True)

        self.dataFilePathLineEdit = QLineEdit()
        self.dataFilePathLineEdit.setReadOnly(True)

        self.selectFileDirectoryLineEdit = QLineEdit()
        self.selectFileDirectoryLineEdit.setReadOnly(True)

        # ---------- Row selector ----------
        RowReminder = QLabel("Row to be optimized:")
        self.rowSelectSpinBox = QSpinBox()
        self.rowSelectSpinBox.valueChanged.connect(self.clearOptContent)
        self.rowSelectSpinBox.valueChanged.connect(self.displaySelectedRow)

        # ---------- Parameter spin box ----------
        self.doubleSpinBox = QDoubleSpinBox()
        self.doubleSpinBox.setMinimum(0.0)
        self.doubleSpinBox.setMaximum(1.0)
        self.doubleSpinBox.setSingleStep(0.01)
        self.doubleSpinBox.setValue(0.8)
        self.doubleSpinBox.valueChanged.connect(self.onDoubleSpinBoxValueChanged)

        # ---------- Layouts ----------
        LoadPKLLayout = QHBoxLayout()
        LoadPKLLayout.addWidget(loadPklButton)
        LoadPKLLayout.addWidget(self.pklFilePathLineEdit)

        LoadXLSXLayout = QHBoxLayout()
        LoadXLSXLayout.addWidget(loadXlsxButton)
        LoadXLSXLayout.addWidget(self.xlsxFilePathLineEdit)

        LoadDataLayout = QHBoxLayout()
        LoadDataLayout.addWidget(loadDataButton)
        LoadDataLayout.addWidget(self.dataFilePathLineEdit)

        SelectFileLayout = QHBoxLayout()
        SelectFileLayout.addWidget(loadLogDirButton)
        SelectFileLayout.addWidget(self.selectFileDirectoryLineEdit)

        runLayout = QHBoxLayout()
        runLayout.addWidget(RowReminder)
        runLayout.addWidget(self.rowSelectSpinBox)
        runLayout.addWidget(RunButton)
        runLayout.addWidget(PlotButton)

        left_layout.addLayout(LoadPKLLayout)
        left_layout.addLayout(LoadXLSXLayout)
        left_layout.addLayout(LoadDataLayout)
        left_layout.addLayout(SelectFileLayout)
        left_layout.addLayout(runLayout)
        left_layout.addLayout(ChangeButtonLayout)

        # ---------- Table ----------
        self.tableWidget = QTableWidget()
        self.tableWidget.clicked.connect(self.onTableCellClicked)
        self.tableWidget.clicked.connect(self.getDatesRange)

        PthreLabel = QLabel("para. f:")
        paralayout = QHBoxLayout()
        paralayout.addWidget(PthreLabel, 1)
        paralayout.addWidget(self.doubleSpinBox, 1)
        paralayout.addWidget(ManualChangeButton, 2)

        left_layout.addWidget(self.tableWidget)
        left_layout.addLayout(paralayout)
        left_layout.addWidget(ViewButton)
        left_layout.addWidget(SaveButton)

        # ---------- Figures ----------
        self.ori_fig, self.ori_axes = plt.subplots(nrows=7, ncols=1, figsize=(4, 14))
        self.FigOriCanvas = FigureCanvas(figure=self.ori_fig)
        OriFigureLayout = QVBoxLayout()
        OriFigureLayout.addWidget(self.FigOriCanvas)

        self.opt_fig, self.opt_axes = plt.subplots(nrows=7, ncols=1, figsize=(4, 14))
        self.FigOptCanvas = FigureCanvas(figure=self.opt_fig)
        OptFigureLayout = QVBoxLayout()
        OptFigureLayout.addWidget(self.FigOptCanvas)

        self.sour_fig, self.sour_axes = plt.subplots(nrows=7, ncols=1, figsize=(4, 14))
        self.FigSourCanvas = FigureCanvas(figure=self.sour_fig)
        SourFigureLayout = QVBoxLayout()
        SourFigureLayout.addWidget(self.FigSourCanvas)

        right_layout.addLayout(OriFigureLayout)
        right_layout.addLayout(OptFigureLayout)
        right_layout.addLayout(SourFigureLayout)

        layout = QHBoxLayout()
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 3)

        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def n_sessions(self) -> int:
        """Return the number of registration/session columns."""
        if self.df_titles is None:
            return 0
        return len(self.df_titles)

    def session_titles(self) -> np.ndarray:
        """Return session titles as a NumPy array to allow array indexing."""
        if self.df_titles is None:
            return np.asarray([])
        return np.asarray(self.df_titles)

    def set_table_headers(self, n_rows: int, column_labels):
        """Safely set vertical and horizontal headers."""
        if self.df is not None:
            vertical_labels = self.df.columns.astype(str).tolist()
            vertical_labels = vertical_labels[:n_rows]
            self.tableWidget.setVerticalHeaderLabels(vertical_labels)

        self.tableWidget.setHorizontalHeaderLabels(column_labels)

    def get_current_original_content(self, index: int) -> np.ndarray:
        """Extract the current selected row across session columns."""
        n = self.n_sessions()
        vals = self.df.iloc[index, :n].copy()
        vals = vals.fillna(0)
        return vals.values.astype(np.int64)

    def add_empty_row_to_df(self):
        """Append one empty row with all existing columns set to 0."""
        new_row = {col: 0 for col in self.df.columns}
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------
    def onDoubleSpinBoxValueChanged(self):
        self._p_thre = self.doubleSpinBox.value()

    def clearOptContent(self):
        self.opt_content = None

    def configureLogger(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not dir_name:
            return

        try:
            self.selectFileDirectoryLineEdit.setText(dir_name)
            file_name = os.path.join(dir_name, "neuromatch_gui.log")

            logging.basicConfig(
                filename=file_name,
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )

            self.log_name = dir_name

        except Exception as e:
            QMessageBox.critical(self, "Directory Load Error", f"An error occurred: {e}")

    def loadPklFile(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open PKL File", "", "Pickle Files (*.pkl)"
        )
        if not file_name:
            return

        try:
            with open(file_name, "rb") as file:
                (
                    self.index_map,
                    self.ref_indexmaps,
                    self.ata_p_sames,
                    self.ata_indexmaps,
                    self.df_titles,
                ) = pickle.load(file)

            # Critical fix:
            # Make df_titles support len(...), integer indexing, and NumPy-array indexing.
            self.df_titles = np.asarray(self.df_titles)

            print(f"{file_name} is loaded successfully!")
            self.pklFilePathLineEdit.setText(file_name)
            QMessageBox.information(self, "File Load", "PKL file loaded successfully.")
            logging.info(f"CellReg Ref Data:\n    {file_name}")

        except Exception as e:
            QMessageBox.critical(self, "File Load Error", f"An error occurred: {e}")

    def loadData(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open PKL File", "", "Pickle Files (*.pkl)"
        )
        if not file_name:
            return

        try:
            with open(file_name, "rb") as file:
                self.data = pickle.load(file)

            print(f"{file_name} is loaded successfully!")
            self.dataFilePathLineEdit.setText(file_name)
            QMessageBox.information(self, "File Load", "PKL file loaded successfully.")

            self._init_data()
            print("  Finish linearized trajectories")
            logging.info(f"Imaging Data:\n    {file_name}")

        except Exception as e:
            QMessageBox.critical(self, "File Load Error", f"An error occurred: {e}")

    def _init_data(self):
        if self.data is None:
            return

        linearized_xs = []
        self.maze_type = self.data["maze_type"][0]

        for k in range(len(self.data["spike_nodes"])):
            spike_nodes = S2F[self.data["spike_nodes"][k].astype(int) - 1]
            linearized_x = np.zeros_like(spike_nodes, dtype=np.float64)
            graph = NRG[int(self.maze_type)]

            for i in range(spike_nodes.shape[0]):
                linearized_x[i] = graph[int(spike_nodes[i])]

            linearized_x = linearized_x + np.random.rand(spike_nodes.shape[0]) - 0.5
            linearized_xs.append(linearized_x)

        self.data["linearized_x"] = linearized_xs

    def loadXlsxFile(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open XLSX File", "", "Excel Files (*.xlsx)"
        )
        if not file_name:
            return

        try:
            xls = pd.ExcelFile(file_name)
            sheet_names = xls.sheet_names

            sheet, ok = QInputDialog.getItem(
                self, "Select Sheet", "Select a sheet to load:", sheet_names, 0, False
            )

            if not (ok and sheet):
                return

            self.df = pd.read_excel(file_name, sheet_name=sheet)
            self._excel_dirname = file_name

            self.displayDataFrame()
            self.updateRowSelectSpinBox()
            self.xlsxFilePathLineEdit.setText(file_name)

            self._ori_n = self.df.shape[0]

            print(f"{file_name} is loaded successfully!")
            QMessageBox.information(
                self, "File Load", f"Sheet '{sheet}' loaded successfully."
            )

            self.setupAutoSaveTimer()

            logging.info(
                f"Excel Data:\n    {file_name}, "
                f"with initial shape {self.df.shape}, and title: {self.df_titles}"
            )

        except Exception as e:
            QMessageBox.critical(self, "File Load Error", f"An error occurred: {e}")

    # ------------------------------------------------------------------
    # Table display
    # ------------------------------------------------------------------
    def displayDataFrame(self):
        if self.df is None:
            return

        self.tableWidget.setRowCount(self.df.shape[1])
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setVerticalHeaderLabels(self.df.columns.astype(str).tolist())
        self.tableWidget.setHorizontalHeaderLabels(["Original"])

    def updateRowSelectSpinBox(self):
        if self.df is not None:
            self.rowSelectSpinBox.setMinimum(0)
            self.rowSelectSpinBox.setMaximum(max(self.df.shape[0] - 1, 0))

    def displaySelectedRow(self, is_comparison: bool = False, is_newtable: bool = True):
        """Display the selected row in the table widget."""
        index = self.rowSelectSpinBox.value()

        if self.df is None:
            print("df is None")
            return

        if self.df_titles is None:
            print("df_titles is None. Please load the PKL file first.")
            return

        n_sessions = self.n_sessions()

        if index >= self.df.shape[0]:
            print(f"index {index} >= self.df.shape[0] {self.df.shape[0]}")
            return

        self.plot_range = None

        if is_comparison and self.opt_content is not None:
            if is_newtable:
                self.tableWidget.setRowCount(n_sessions + 2)

                for i in range(n_sessions + 2):
                    self.tableWidget.setRowHeight(i, 20)

                self.tableWidget.setColumnCount(4)
                for j in range(4):
                    self.tableWidget.setColumnWidth(j, 60)

                self.set_table_headers(
                    n_rows=n_sessions + 2,
                    column_labels=["Origi.", "Optim.", "Source", "Replace"],
                )

            self.ori_content = self.get_current_original_content(index)
            self.df.iloc[index, n_sessions] = np.count_nonzero(self.ori_content)

            newfound_idx = np.where(
                (self.ori_content - self.opt_content != 0) & (self.ori_content == 0)
            )[0]
            adjusted_idx = np.where(
                (self.ori_content - self.opt_content != 0) & (self.ori_content != 0)
            )[0]

            for j in range(min(self.df.shape[1], n_sessions + 2)):
                value = self.df.iloc[index, j]
                if pd.isna(value):
                    value = 0
                    self.df.iloc[index, j] = 0

                item = QTableWidgetItem(str(int(value)))

                if j in newfound_idx:
                    item.setBackground(QColor(255, 255, 224))  # light yellow
                if j in adjusted_idx:
                    item.setBackground(QColor(216, 191, 216))  # light purple

                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tableWidget.setItem(j, 0, item)

            for j in range(self.opt_content.shape[0]):
                item = QTableWidgetItem(str(int(self.opt_content[j])))

                if j in newfound_idx:
                    item.setBackground(QColor(173, 216, 230))  # light blue
                if j in adjusted_idx:
                    item.setBackground(QColor(144, 238, 144))  # light green

                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tableWidget.setItem(j, 1, item)

            item = QTableWidgetItem(str(np.count_nonzero(self.opt_content)))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tableWidget.setItem(self.opt_content.shape[0], 1, item)

        else:
            self.tableWidget.setRowCount(n_sessions + 2)

            for i in range(n_sessions + 2):
                self.tableWidget.setRowHeight(i, 20)

            self.tableWidget.setColumnCount(1)
            self.tableWidget.setColumnWidth(0, 60)

            self.set_table_headers(
                n_rows=n_sessions + 2,
                column_labels=["Original"],
            )

            self.ori_content = self.get_current_original_content(index)
            self.df.iloc[index, n_sessions] = np.count_nonzero(self.ori_content)

            for j in range(min(self.df.shape[1], n_sessions + 2)):
                value = self.df.iloc[index, j]
                if pd.isna(value):
                    value = 0
                    self.df.iloc[index, j] = 0

                item = QTableWidgetItem(str(int(value)))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tableWidget.setItem(j, 0, item)

    # ------------------------------------------------------------------
    # Core optimization
    # ------------------------------------------------------------------
    def run(self):
        if self.ori_content is None:
            QMessageBox.warning(self, "Run Error", "Please select a row first.")
            return

        index = self.rowSelectSpinBox.value()
        self.ori_content = self.get_current_original_content(index)

        if np.where(self.ori_content != 0)[0].shape[0] <= 1:
            QMessageBox.warning(
                self,
                "Warning",
                "Cannot compute optimized neuron pair because there is only 1 neuron!",
            )
            return

        try:
            reg_neuron = RegisteredNeuron(
                index_line=self.ori_content,
                ref_indexmaps=self.ref_indexmaps,
                ata_p_sames=AllToAllList(self.ata_p_sames),
                ata_indexmaps=AllToAllList(self.ata_indexmaps),
                p_thre=self._p_thre,
            )

            print(f"Row {self.rowSelectSpinBox.value()}")
            print(reg_neuron.ori_content)

            reg_neuron.optimize()
            self.opt_content = reg_neuron.opt_content

            print(reg_neuron.opt_content)
            self.displaySelectedRow(is_comparison=True)

        except Exception as e:
            QMessageBox.warning(
                self,
                "Error During Optimizing",
                f"An error was thrown: {e}. Most likely because of division by zero.",
            )

    # ------------------------------------------------------------------
    # Interaction with table
    # ------------------------------------------------------------------
    def displaySourceRow(self):
        if self.sour_content is None:
            return

    def onTableCellClicked(self, qindex: QModelIndex):
        row, column = qindex.row(), qindex.column()
        self._row, self._col = row, column

        n_sessions = self.n_sessions()

        if column != 1:
            return

        if row >= n_sessions:
            return

        if self.opt_content is None:
            return

        try:
            index = np.where(self.df.iloc[:, row] == self.opt_content[row])[0][0]
        except Exception as e:
            QMessageBox.critical(
                self,
                "Index Error",
                f"An error occurred: {e}.\n"
                f"This error was raised when {self.opt_content[row]} could not be found "
                f"in column {row}, entitled {self.df_titles[row]}",
            )
            return

        self.sour_content = self.df.iloc[index, :n_sessions].fillna(0).values.astype(np.int64)

        print(f" Click on Item {row}, {column}, with value {self.sour_content[row]}")

        newfound_idx = np.where(
            (self.ori_content - self.opt_content != 0) & (self.ori_content == 0)
        )[0]
        adjusted_idx = np.where(
            (self.ori_content - self.opt_content != 0) & (self.ori_content != 0)
        )[0]

        for j in range(self.sour_content.shape[0]):
            item = QTableWidgetItem(str(int(self.sour_content[j])))

            if j in newfound_idx and j == row:
                item.setBackground(QColor(173, 216, 230))  # light blue
            if j in adjusted_idx and j == row:
                item.setBackground(QColor(144, 238, 144))  # light green

            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tableWidget.setItem(j, 2, item)

        item = QTableWidgetItem(str(np.count_nonzero(self.sour_content)))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.tableWidget.setItem(self.sour_content.shape[0], 2, item)

        item = QTableWidgetItem(str(index))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.tableWidget.setItem(self.sour_content.shape[0] + 1, 2, item)

    def getDatesRange(self, qindex: QModelIndex):
        row, column = qindex.row(), qindex.column()
        self._row, self._col = row, column

        titles = self.session_titles()
        n_sessions = len(titles)

        if row >= n_sessions:
            return

        if row + 3 >= n_sessions:
            start = max(n_sessions - 7, 0)
            self.plot_range = np.arange(start, n_sessions)
        elif row - 3 < 0:
            stop = min(7, n_sessions)
            self.plot_range = np.arange(stop)
        else:
            self.plot_range = np.arange(row - 3, row + 4)

        print(" Plot range of dates (7 sessions):", titles[self.plot_range])

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def plot(self):
        if self.plot_range is None:
            QMessageBox.warning(
                self,
                "Plot Figures",
                "Please select the item to be plotted from the sheet first!",
            )
            return

        if self.data is None:
            QMessageBox.warning(self, "Plot Figures", "Please load imaging data first!")
            return

        if self.ori_content is not None:
            LocTimeCurve(
                self.data,
                self.ori_axes,
                self.plot_range,
                self.ori_content[self.plot_range],
                line_kwargs={"markeredgewidth": 0, "markersize": 0.8, "color": "gray"},
                bar_kwargs={"markeredgewidth": 1, "markersize": 4},
            )
            self.FigOriCanvas.draw()
            print("Plotting:", self.ori_content[self.plot_range])

        if self.opt_content is not None:
            LocTimeCurve(
                self.data,
                self.opt_axes,
                self.plot_range,
                self.opt_content[self.plot_range],
                line_kwargs={"markeredgewidth": 0, "markersize": 0.8, "color": "gray"},
                bar_kwargs={"markeredgewidth": 1, "markersize": 4},
            )
            self.FigOptCanvas.draw()
            print("Plotting:", self.opt_content[self.plot_range])

        if self.sour_content is not None:
            LocTimeCurve(
                self.data,
                self.sour_axes,
                self.plot_range,
                self.sour_content[self.plot_range],
                line_kwargs={"markeredgewidth": 0, "markersize": 0.8, "color": "gray"},
                bar_kwargs={"markeredgewidth": 1, "markersize": 4},
            )
            self.FigSourCanvas.draw()
            print("Plotting:", self.sour_content[self.plot_range])

    # ------------------------------------------------------------------
    # Editing operations
    # ------------------------------------------------------------------
    def fill(self):
        if self._row is None or self._col is None:
            return

        if self.opt_content is None:
            return

        n_sessions = self.n_sessions()

        if self._row >= n_sessions:
            QMessageBox.warning(
                self,
                "Operation Warning",
                f"Please click on a valid item to identify where the position should be filled, "
                f"instead of row {self._row + 1}.",
            )
            return

        i, j = self.rowSelectSpinBox.value(), self._row

        reply = QMessageBox.question(
            self,
            "Confirm Action",
            f"Are you sure to fill the vacancy at row {j + 1} with Cell {self.opt_content[j]}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            index = np.where(self.df.iloc[:, j] == self.opt_content[j])[0][0]
        except Exception as e:
            QMessageBox.warning(
                self,
                "Operation Error",
                f"Error: {e}. Do not find {self.opt_content[j]} in row {j + 1}",
            )
            return

        if index == i:
            return

        if self.df.iloc[i, j] != 0:
            QMessageBox.information(self, "Update Warning", "Please push button Replace in this case.")
            return

        logging.info(f"Excel Row {i} adopted a change, on Session {j + 1}: {self.df_titles[j]}")
        logging.info(f"    Original: {self.ori_content}")
        logging.info(f"    Optimized: {self.opt_content}")
        logging.info(f"    Changes: {0} was filled with Cell {self.opt_content[j]}")
        logging.info(
            f"  - Excel Row {index} was coordinately changed by deleting Cell {self.opt_content[j]}"
        )
        logging.info(f"    from {self.df.iloc[index, :n_sessions].astype(int)}\n\n")

        print(f"fill the vacancy at row {j} with Cell {self.opt_content[j]}")

        self.df.iloc[index, j] = 0
        self.df.iloc[i, j] = self.opt_content[j]

        item = QTableWidgetItem(str(int(self.opt_content[j])))
        item.setBackground(QColor(255, 182, 193))  # light red
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.tableWidget.setItem(j, 0, item)

        self.displaySelectedRow(is_comparison=True, is_newtable=False)
        print("Fill Success.")

    def replace(self):
        if self._row is None or self._col is None:
            return

        if self.opt_content is None:
            return

        n_sessions = self.n_sessions()

        if self._row >= n_sessions:
            QMessageBox.warning(
                self,
                "Operation Warning",
                f"Please click on a valid item to identify where the position should be filled, "
                f"instead of row {self._row + 1}.",
            )
            return

        i, j = self.rowSelectSpinBox.value(), self._row

        reply = QMessageBox.question(
            self,
            "Confirm Action",
            f"Are you sure to replace Cell {self.ori_content[j]} at row {j + 1} "
            f"with Cell {self.opt_content[j]}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        print(f"replace Cell {self.ori_content[j]} at row {j} with Cell {self.opt_content[j]}")

        try:
            index = np.where(self.df.iloc[:, j] == self.opt_content[j])[0][0]
        except Exception as e:
            QMessageBox.warning(
                self,
                "Operation Error",
                f"Error: {e}. Do not find {self.opt_content[j]} in row {j + 1}",
            )
            return

        if self.df.iloc[i, j] == 0:
            QMessageBox.information(self, "Update Warning", "Please push button Fill in this case.")
            return

        if index == i:
            return

        if self._ori_n is None:
            self._ori_n = self.df.shape[0]

        logging.info(f"Excel Row {i + 1} adopted a change, on Session {j + 1}: {self.df_titles[j]}")
        logging.info(f"    Original: {self.ori_content}")
        logging.info(f"    Optimized: {self.opt_content}")
        logging.info(f"    Changes: {self.ori_content[j]} was replaced by Cell {self.opt_content[j]}")
        logging.info(
            f"  - Excel Row {index} was coordinately changed by deleting Cell {self.opt_content[j]}"
        )
        logging.info(f"    from {self.df.iloc[index, :n_sessions].astype(int)}")
        logging.info(f"  - Replaced Cell {self.ori_content[j]} was moved to line {self._ori_n}:")

        self.df.iloc[index, j] = 0
        self.df.iloc[i, j] = self.opt_content[j]

        item = QTableWidgetItem(str(int(self.opt_content[j])))
        item.setBackground(QColor(173, 216, 230))  # light blue
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.tableWidget.setItem(j, 0, item)

        moved_row_idx = self.df.shape[0]
        self.add_empty_row_to_df()
        self.df.iloc[moved_row_idx, j] = self.ori_content[j]

        for k in range(n_sessions):
            value = self.df.iloc[moved_row_idx, k]
            if pd.isna(value):
                value = 0
                self.df.iloc[moved_row_idx, k] = 0

            item = QTableWidgetItem(str(int(value)))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tableWidget.setItem(k, 3, item)

        self.df.iloc[moved_row_idx, n_sessions] = np.count_nonzero(
            self.df.iloc[moved_row_idx, :n_sessions]
        )

        item = QTableWidgetItem(str(int(self.df.iloc[moved_row_idx, n_sessions])))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.tableWidget.setItem(n_sessions, 3, item)

        logging.info(f"      {self.df.iloc[moved_row_idx, :n_sessions]}\n\n")

        self.displaySelectedRow(is_comparison=True, is_newtable=False)
        print("Replace Success.")

    def moveOut(self, i=None, j=None):
        if self._row is None or self._col is None:
            return

        if self.opt_content is None:
            return

        n_sessions = self.n_sessions()

        if self._row >= n_sessions:
            QMessageBox.warning(
                self,
                "Operation Warning",
                f"Please click on a valid item to identify where the position should be filled, "
                f"instead of row {self._row + 1}.",
            )
            return

        i, j = self.rowSelectSpinBox.value(), self._row

        reply = QMessageBox.question(
            self,
            "Confirm Action",
            f"Are you sure to move out Cell {self.ori_content[j]} at row {j + 1}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        print(f"Move out Cell {self.ori_content[j]} at row {j}")

        if self.df.iloc[i, j] == 0:
            QMessageBox.information(self, "Update Warning", f"There is nothing to move out at row {i}")
            return

        logging.info(f"Excel Row {i} adopted a change, on Session {j + 1}: {self.df_titles[j]}")
        logging.info(f"    Original: {self.ori_content}")
        logging.info(f"    Optimized: {self.opt_content}")
        logging.info(f"    Changes: {self.ori_content[j]} was moved out")

        self.df.iloc[i, j] = 0

        item = QTableWidgetItem("0")
        item.setBackground(QColor(173, 216, 230))  # light blue
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.tableWidget.setItem(j, 0, item)

        moved_row_idx = self.df.shape[0]
        self.add_empty_row_to_df()
        self.df.iloc[moved_row_idx, j] = self.ori_content[j]

        for k in range(n_sessions):
            value = self.df.iloc[moved_row_idx, k]
            if pd.isna(value):
                value = 0
                self.df.iloc[moved_row_idx, k] = 0

            item = QTableWidgetItem(str(int(value)))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tableWidget.setItem(k, 3, item)

        self.df.iloc[moved_row_idx, n_sessions] = np.count_nonzero(
            self.df.iloc[moved_row_idx, :n_sessions]
        )

        item = QTableWidgetItem(str(int(self.df.iloc[moved_row_idx, n_sessions])))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.tableWidget.setItem(n_sessions, 3, item)

        logging.info(f"      {self.df.iloc[moved_row_idx, :n_sessions]}\n\n")

        self.displaySelectedRow(is_comparison=True, is_newtable=False)
        print("Move out Success.")

    def manualChange(self):
        if self._row is None or self._col is None:
            return

        if self.opt_content is None:
            QMessageBox.warning(self, "Operation Warning", "Please run optimization first!")
            return

        n_sessions = self.n_sessions()

        if self._row >= n_sessions:
            QMessageBox.warning(
                self,
                "Operation Warning",
                f"Please click on a valid item to identify where the position should be filled, "
                f"instead of row {self._row + 1}.",
            )
            return

        num, ok = QInputDialog.getInt(self, "Input Number", "Enter a number:")

        if ok and num is not None:
            self.opt_content[self._row] = num

            if self.df.iloc[self.rowSelectSpinBox.value(), self._row] == 0:
                self.fill()
            else:
                self.replace()

    def adopt(self):
        if self.df is None:
            return

        self._ori_n = self.df.shape[0]

        next_row = self.rowSelectSpinBox.value() + 1
        if next_row <= self.rowSelectSpinBox.maximum():
            self.rowSelectSpinBox.setValue(next_row)

        self._row, self._col = None, None
        self.opt_content, self.sour_content = None, None

    # ------------------------------------------------------------------
    # View and save
    # ------------------------------------------------------------------
    def view(self):
        if self.df is not None:
            self.dataFrameViewer = DataFrameViewer(self.df, parent=self)
            self.dataFrameViewer.show()

    def setupAutoSaveTimer(self):
        if self.autoSaveTimer is not None:
            self.autoSaveTimer.stop()

        self.autoSaveTimer = QTimer(self)
        self.autoSaveTimer.timeout.connect(self.autoSaveData)
        self.autoSaveTimer.start(180000)

    def autoSaveData(self):
        self.save(is_autosave=True)

    def save(self, is_autosave: bool = False):
        if self.df is None:
            QMessageBox.information(self, "File Save", "No DataFrame to save.")
            return

        if self.log_name is None:
            QMessageBox.information(self, "File Save", "Please select a directory first!")
            return

        if self._excel_dirname is None:
            QMessageBox.information(self, "File Save", "Please load an Excel file first!")
            return

        n_sessions = self.n_sessions()

        try:
            xlsx_path = os.path.join(os.path.dirname(self._excel_dirname), "neuromatch_res.xlsx")
            pkl_path = os.path.join(os.path.dirname(self._excel_dirname), "neuromatch_res.pkl")

            self.df.to_excel(xlsx_path, sheet_name="data", index=False)

            index_map = [self.df.iloc[:, i].fillna(0).values for i in range(n_sessions)]
            index_map = np.vstack(index_map)

            mat = np.where(index_map > 0, 1, 0)
            num = np.sum(mat, axis=0)
            idx = np.where(num > 0)[0]

            with open(pkl_path, "wb") as f:
                pickle.dump(index_map[:, idx], f)

            if not is_autosave:
                QMessageBox.information(self, "File Save", f"File saved successfully ({xlsx_path})")
                print("File saved successfully!")
                print("    File path:", xlsx_path)
                print("    File path:", pkl_path)

        except Exception as e:
            QMessageBox.critical(self, "File Save Error", f"An error occurred: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = NeuroMatchGUI()
    mainWindow.show()
    sys.exit(app.exec())