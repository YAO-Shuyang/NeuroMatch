import sys
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QTableWidget
from PyQt6.QtWidgets import QTableWidgetItem, QFileDialog, QMessageBox
from PyQt6.QtWidgets import QInputDialog, QLineEdit, QSpinBox, QLabel
from PyQt6.QtCore import Qt, QModelIndex, QTimer
from PyQt6.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from neuromatch.variables.regneuron import RegisteredNeuron
from neuromatch.variables import AllToAllList
from neuromatch.visualize.loctime import LocTimeCurve
from neuromatch.gui.viewer import DataFrameViewer

from neuromatch.visualize.maze_graph import NRG, S2F

class NeuroMatchGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroMatch GUI v1.0 beta")
        self.setGeometry(100, 100, 1200, 600)
        self.initUI()

    def initUI(self):
        # Create layout
        Leftlayout = QVBoxLayout()
        RightLayout = QHBoxLayout()
        self.opt_content = None
        self.ori_content = None
        self.df_titles = None
        self.df = None
        self.data = None
        self.sour_content = None
        self.plot_range = None
        self._row, self._col = None, None

        # Create buttons
        loadPklButton = QPushButton("Load PKL File")
        loadXlsxButton = QPushButton("Load Excel File")
        loadPklButton.clicked.connect(self.loadPklFile)
        loadXlsxButton.clicked.connect(self.loadXlsxFile)
        loadDataButton = QPushButton("Load Data")
        loadDataButton.clicked.connect(self.loadData)
        RunButton = QPushButton("Run")
        RunButton.clicked.connect(self.run)
        PlotButton = QPushButton("Plot")
        PlotButton.clicked.connect(self.plot)
        
        # replace
        FillButton = QPushButton("Fill")
        FillButton.clicked.connect(self.fill)
        ReplaceButton = QPushButton("Replace")
        ReplaceButton.clicked.connect(self.replace)
        AdoptButton = QPushButton("Adopt")
        AdoptButton.clicked.connect(self.adopt)
        
        ChangeButtonLayout = QHBoxLayout()
        ChangeButtonLayout.addWidget(FillButton)
        ChangeButtonLayout.addWidget(ReplaceButton)
        ChangeButtonLayout.addWidget(AdoptButton)
        
        # save
        ViewButton = QPushButton("View")
        ViewButton.clicked.connect(self.view)
        SaveButton = QPushButton("Save")
        SaveButton.clicked.connect(self.save)

        # Create text widgets for displaying file paths
        self.pklFilePathLineEdit = QLineEdit()
        self.pklFilePathLineEdit.setReadOnly(True)  # Make it read-only
        self.xlsxFilePathLineEdit = QLineEdit()
        self.xlsxFilePathLineEdit.setReadOnly(True)  # Make it read-only
        self.dataFilePathLineEdit = QLineEdit()
        self.dataFilePathLineEdit.setReadOnly(True)  # Make it read-only
        
        # Add a spin box for selecting rows
        RowReminder = QLabel("Row to be optimized:")
        self.rowSelectSpinBox = QSpinBox()
        self.rowSelectSpinBox.valueChanged.connect(self.clearOptContent)
        self.rowSelectSpinBox.valueChanged.connect(self.displaySelectedRow)
        
        # Button layout
        LoadPKLLayout = QHBoxLayout()
        LoadPKLLayout.addWidget(loadPklButton)
        LoadPKLLayout.addWidget(self.pklFilePathLineEdit)
        LoadXLSXLayout = QHBoxLayout()
        LoadXLSXLayout.addWidget(loadXlsxButton)
        LoadXLSXLayout.addWidget(self.xlsxFilePathLineEdit)
        LoadDataLayout = QHBoxLayout()
        LoadDataLayout.addWidget(loadDataButton)
        LoadDataLayout.addWidget(self.dataFilePathLineEdit)
        
        # RunLayout
        runLayout = QHBoxLayout()
        runLayout.addWidget(RowReminder)
        runLayout.addWidget(self.rowSelectSpinBox)
        runLayout.addWidget(RunButton)
        runLayout.addWidget(PlotButton)
        
        # Add button layout to main layout
        Leftlayout.addLayout(LoadPKLLayout)
        Leftlayout.addLayout(LoadXLSXLayout)
        Leftlayout.addLayout(LoadDataLayout)
        Leftlayout.addLayout(runLayout)
        Leftlayout.addLayout(ChangeButtonLayout)
        

        # Create table for displaying XLSX content
        self.tableWidget = QTableWidget()
        self.tableWidget.clicked.connect(self.onTableCellClicked)
        self.tableWidget.clicked.connect(self.getDatesRange)
        Leftlayout.addWidget(self.tableWidget)
        Leftlayout.addWidget(ViewButton)
        Leftlayout.addWidget(SaveButton)
        
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
        RightLayout.addLayout(OriFigureLayout)
        RightLayout.addLayout(OptFigureLayout)
        RightLayout.addLayout(SourFigureLayout)
        

        layout = QHBoxLayout()
        layout.addLayout(Leftlayout, 1)
        layout.addLayout(RightLayout, 3)
        # Set the layout to the central widget
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)


    def clearOptContent(self):
        self.opt_content = None

    def loadPklFile(self):
        # Open a file dialog to select the .pkl file
        file_name, _ = QFileDialog.getOpenFileName(self, "Open PKL File", "", "Pickle Files (*.pkl)")
        if file_name:
            try:
                with open(file_name, 'rb') as file:
                    self.index_map, self.ref_indexmaps, self.ata_p_sames, self.ata_indexmaps, self.df_titles = pickle.load(file)
                    print(f"{file_name} is loaded successfully!")
                self.pklFilePathLineEdit.setText(file_name)
                QMessageBox.information(self, "File Load", "PKL file loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "File Load Error", f"An error occurred: {e}")
                
    def loadData(self):
        # Open a file dialog to select the .pkl file
        file_name, _ = QFileDialog.getOpenFileName(self, "Open PKL File", "", "Pickle Files (*.pkl)")
        if file_name:
            try:
                with open(file_name, 'rb') as file:
                    self.data = pickle.load(file)
                    print(f"{file_name} is loaded successfully!")
                self.dataFilePathLineEdit.setText(file_name)
                QMessageBox.information(self, "File Load", "PKL file loaded successfully.")
                self._init_data()
                print("  Finish linearized trajectories")
            except Exception as e:
                QMessageBox.critical(self, "File Load Error", f"An error occurred: {e}")     
                
    def _init_data(self):
        if self.data is None:
            return
        
        linearized_xs = []
        self.maze_type = self.data['maze_type'][0]
        for i in range(len(self.data['spike_nodes'])):
            spike_nodes = S2F[self.data['spike_nodes'][i].astype(int)-1]
            linearized_x = np.zeros_like(spike_nodes, np.float64)
            graph = NRG[int(self.maze_type)]

            for i in range(spike_nodes.shape[0]):
                linearized_x[i] = graph[int(spike_nodes[i])]
    
            linearized_x = linearized_x + np.random.rand(spike_nodes.shape[0]) - 0.5
            linearized_xs.append(linearized_x)
            
        self.data['linearized_x'] = linearized_xs
        
    def loadXlsxFile(self):
        # Open a file dialog to select the .xlsx file
        file_name, _ = QFileDialog.getOpenFileName(self, "Open XLSX File", "", "Excel Files (*.xlsx)")
        if file_name:
            try:
                # Get the list of sheet names
                xls = pd.ExcelFile(file_name)
                sheet_names = xls.sheet_names

                # Ask the user to select a sheet
                sheet, ok = QInputDialog.getItem(self, "Select Sheet", "Select a sheet to load:", sheet_names, 0, False)
                if ok and sheet:
                    # Load the selected sheet
                    self.df = pd.read_excel(file_name, sheet_name=sheet)
                    self.displayDataFrame()
                    QMessageBox.information(self, "File Load", f"Sheet '{sheet}' loaded successfully.")
                    self.xlsxFilePathLineEdit.setText(file_name)
                    self._excel_dirname = file_name
                    self.updateRowSelectSpinBox()
                    print(f"{file_name} is loaded successfully!")
                    self._ori_n = self.df.shape[0] # The original length of the excel sheet.
                    self.setupAutoSaveTimer()
                    
            except Exception as e:
                QMessageBox.critical(self, "File Load Error", f"An error occurred: {e}")

    def displayDataFrame(self):
        #self.tableWidget.setRowCount(df.shape[0])
        #self.tableWidget.setColumnCount(df.shape[1])
        self.tableWidget.setRowCount(self.df.shape[1])
        self.tableWidget.setVerticalHeaderLabels(self.df.columns.astype(str))

    
    def updateRowSelectSpinBox(self):
        if self.df is not None:
            self.rowSelectSpinBox.setMaximum(self.df.shape[0] - 1)
        """Update the row selection spin box with the number of rows in the DataFrame."""

    def displaySelectedRow(self, is_comparison: bool = False):
        """Display the selected row in the table widget."""
        index = self.rowSelectSpinBox.value()
        if self.df is None or self.df_titles is None:
            print(f"df {self.df} or df_titles {self.df_titles} is None")
            return
        
        if index >= self.df.shape[0]:
            print(f"index {index} >= self.df.shape[0] {self.df.shape[0]}")
            return
        
        self.plot_range = None
        
        if is_comparison and self.opt_content is not None:
            row_data = self.df.iloc[index]
            self.tableWidget.setRowCount(self.df.shape[1])
            self.tableWidget.setColumnCount(4)
            self.tableWidget.setColumnWidth(0, 60)
            self.tableWidget.setColumnWidth(1, 60)
            self.tableWidget.setColumnWidth(2, 60)
            self.tableWidget.setColumnWidth(3, 60)
            self.tableWidget.setVerticalHeaderLabels(self.df.columns.astype(str))
            self.tableWidget.setHorizontalHeaderLabels(np.array(["Origi.", "Optim.", "Source", "Replace"]))

            newfound_idx = np.where((self.ori_content - self.opt_content != 0)&(self.ori_content == 0))[0]
            adjusted_idx = np.where((self.ori_content - self.opt_content != 0)&(self.ori_content != 0))[0]
            for j in range(self.df.shape[1]):
                item_value = str(int(row_data.iloc[j]))
                item = QTableWidgetItem(item_value)
                if j in newfound_idx:
                    item.setBackground(QColor(255, 255, 224)) # light yellow
                if j in adjusted_idx:
                    item.setBackground(QColor(216, 191, 216)) # light purple
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tableWidget.setItem(j, 0, item)
                

            for j in range(self.opt_content.shape[0]):
                item_value = str(int(self.opt_content[j]))
                item = QTableWidgetItem(item_value)
                if j in newfound_idx:
                    item.setBackground(QColor(173, 216, 230)) # light blue
                if j in adjusted_idx:
                    item.setBackground(QColor(144, 238, 144)) # light green

                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tableWidget.setItem(j, 1, item)
                
            item = QTableWidgetItem(str(np.count_nonzero(self.opt_content)))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tableWidget.setItem(self.opt_content.shape[0], 1, item)
        else:
            row_data = self.df.iloc[index]
            self.tableWidget.setRowCount(self.df.shape[1])
            self.tableWidget.setColumnCount(1)
            self.tableWidget.setColumnWidth(0, 60)
            self.tableWidget.setVerticalHeaderLabels(self.df.columns.astype(str))
            self.tableWidget.setHorizontalHeaderLabels(["Original"])

            self.ori_content = self.df.iloc[index, :len(self.df_titles)].values.astype(np.int64)
            for j in range(self.df.shape[1]):
                item_value = str(int(row_data.iloc[j]))
                item = QTableWidgetItem(item_value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tableWidget.setItem(j, 0, item)

    def run(self):
        if self.ori_content is None:
            QMessageBox.warning(self, "Run Error", f"Please select a row first.")
            return
        
        reg_neuron = RegisteredNeuron(
            index_line=self.ori_content, 
            ref_indexmaps=self.ref_indexmaps,
            ata_p_sames=AllToAllList(self.ata_p_sames), 
            ata_indexmaps=AllToAllList(self.ata_indexmaps)
        )
        print(f"Row {self.rowSelectSpinBox.value()}")
        print(reg_neuron.ori_content)
        reg_neuron.optimize()
        self.opt_content = reg_neuron.opt_content
        print(reg_neuron.opt_content)
        self.displaySelectedRow(is_comparison=True)

    def displaySourceRow(self):
        if self.sour_content is None:
            return
    
    def onTableCellClicked(self, qindex: QModelIndex):
        row, column = qindex.row(), qindex.column()
        
        if column != 1:
            return
        self._row, self._col = row, column
        
        index = np.where(self.df[self.df_titles[row]] == self.opt_content[row])[0][0]
        self.sour_content = self.df.iloc[index, :len(self.df_titles)].values.astype(np.int64)
        print(f" Click on Item {row}, {column}, with value {self.sour_content[row]}")
        newfound_idx = np.where((self.ori_content - self.opt_content != 0)&(self.ori_content == 0))[0]
        adjusted_idx = np.where((self.ori_content - self.opt_content != 0)&(self.ori_content != 0))[0]
        for j in range(self.sour_content.shape[0]):
            item_value = str(int(self.sour_content[j]))
            item = QTableWidgetItem(item_value)
            if j in newfound_idx and j == row:
                item.setBackground(QColor(173, 216, 230)) # light blue
            if j in adjusted_idx and j == row:
                item.setBackground(QColor(144, 238, 144)) # light green

            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tableWidget.setItem(j, 2, item)
            
        item = QTableWidgetItem(str(np.count_nonzero(self.sour_content)))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.tableWidget.setItem(self.sour_content.shape[0], 2, item)
                
    def getDatesRange(self, qindex: QModelIndex):
        row, column = qindex.row(), qindex.column()
        
        if row + 3 >= len(self.df_titles):
            self.plot_range = np.arange(len(self.df_titles)-7, len(self.df_titles)) if len(self.df_titles)-7 >= 0 else np.arange(len(self.df_titles))
            print(" Plot range of dates (7 sessions):", self.df_titles[self.plot_range])
        elif row - 3 < 0:
            self.plot_range = np.arange(7) if len(self.df_titles) >= 7 else np.arange(len(self.df_titles))
            print(" Plot range of dates (7 sessions):", self.df_titles[self.plot_range])
        else:
            self.plot_range = np.arange(row-3, row+4)
            print(" Plot range of dates (7 sessions):", self.df_titles[self.plot_range])
            

    def plot(self):
        if self.plot_range is None:
            QMessageBox.warning(self, "Plot Figures", f"Please selecte the item to be ploted from the sheet at first!")
            return
        
        if self.ori_content is not None:
            LocTimeCurve(self.data, self.ori_axes, self.plot_range, self.ori_content[self.plot_range],
                         line_kwargs = {'markeredgewidth': 0, 'markersize': 0.8, 'color': 'black'},
                         bar_kwargs={'markeredgewidth': 1, 'markersize': 4})
            self.FigOriCanvas.draw()
            print("Plotting:", self.ori_content[self.plot_range])
        if self.opt_content is not None:
            LocTimeCurve(self.data, self.opt_axes, self.plot_range, self.opt_content[self.plot_range],
                         line_kwargs = {'markeredgewidth': 0, 'markersize': 0.8, 'color': 'black'},
                         bar_kwargs={'markeredgewidth': 1, 'markersize': 4})
            self.FigOptCanvas.draw()
            print("Plotting:", self.opt_content[self.plot_range])
        if self.sour_content is not None:
            LocTimeCurve(self.data, self.sour_axes, self.plot_range, self.sour_content[self.plot_range],
                         line_kwargs = {'markeredgewidth': 0, 'markersize': 0.8, 'color': 'black'},
                         bar_kwargs={'markeredgewidth': 1, 'markersize': 4})
            self.FigSourCanvas.draw()
            print("Plotting:", self.sour_content[self.plot_range])
            
    def fill(self):
        if self._row is None or self._col is None:
            return
        
        if self.opt_content is None:
            return
        
        
        i, j = self.rowSelectSpinBox.value(), self._row
        
        reply = QMessageBox.question(self, "Confirm Action", f"Are you sure to fill the vacancy at row {j} with Cell {self.opt_content[j]}?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply:
            print(f"fill the vacancy at row {j} with Cell {self.opt_content[j]}")
            index = np.where(self.df[self.df_titles[j]] == self.opt_content[j])[0][0]
            self.df.loc[index, j] = 0
            self.df.loc[i, j] = self.opt_content[j]
            item_value = str(int(self.opt_content[j]))
            item = QTableWidgetItem(item_value)
            item.setBackground(QColor(255, 182, 193)) # light red

            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tableWidget.setItem(j, 0, item)
            
        self._row, self._col = None, None
            
    
    def replace(self):
        if self._row is None or self._col is None:
            return
        
        if self.opt_content is None:
            return
        
        i, j = self.rowSelectSpinBox.value(), self._row
        
        reply = QMessageBox.question(self, "Confirm Action", f"Are you sure to replace Cell {self.ori_content[j]} at row {j} with Cell {self.opt_content[j]}?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply:
            print(f"replace the Cell {self.ori_content[j]} at row {j} with Cell {self.opt_content[j]}")
            index = np.where(self.df[self.df_titles[j]] == self.opt_content[j])[0][0]
            self.df.iloc[index, j] = 0
            self.df.iloc[i, j] = self.opt_content[j]
            item_value = str(int(self.opt_content[j]))
            item = QTableWidgetItem(item_value)
            item.setBackground(QColor(173, 216, 230)) # light blue

            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tableWidget.setItem(j, 0, item)
            
            new_row = {}
            for tit in self.df_titles:
                new_row[tit] = 0
            self.df = self.df.append(new_row, ignore_index=True)
            self.df.iloc[self._ori_n, j] = self.ori_content[j]
            for k in range(self.df_titles.shape[0]):
                if np.isnan(self.df.iloc[self._ori_n, k]):
                    self.df.iloc[self._ori_n, k] = 0

                item = QTableWidgetItem(str(int(self.df.iloc[self._ori_n, k])))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tableWidget.setItem(k, 3, item)
            self.df.iloc[self._ori_n, self.df_titles.shape[0]] = np.count_nonzero(self.df.iloc[self._ori_n, :self.df_titles.shape[0]])
            item = QTableWidgetItem(str(int(self.df.iloc[self._ori_n, self.df_titles.shape[0]])))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tableWidget.setItem(k, 3, item)            
        self._row, self._col = None, None
    
    def adopt(self):
        self._ori_n = self.df.shape[0]
        self.rowSelectSpinBox.setValue(self.rowSelectSpinBox.value() + 1)
        self._row, self._col = None, None
        self.opt_content, self.sour_content = None, None
    
    def view(self):
        if self.df is not None:
            # Open the DataFrame viewer window
            self.dataFrameViewer = DataFrameViewer(self.df)
            self.dataFrameViewer.show()

    def setupAutoSaveTimer(self):
        # Set up a timer to trigger every 5 minutes (300000 milliseconds)
        self.autoSaveTimer = QTimer(self)
        self.autoSaveTimer.timeout.connect(self.autoSaveData)
        self.autoSaveTimer.start(300000)
        
    def autoSaveData(self):
        self.save(is_autosave=True)

    def save(self, is_autosave: bool = False):
        try:
            self.df.to_excel(os.path.join(os.path.dirname(self._excel_dirname), "neuromatch_res.xlsx"), sheet_name='data', index=False)
            index_map = [self.df.loc[:, day] for day in self.df_titles]
            index_map = np.vstack(index_map)
            with open(os.path.join(os.path.dirname(self._excel_dirname), "neuromatch_res.pkl"), 'wb') as f:
                pickle.dump(index_map, f)
            
            if is_autosave == False:
                QMessageBox.information(self, "File Save", f"File saved successfully ({os.path.join(os.path.dirname(self._excel_dirname), 'neuromatch_res.xlsx')})")
                print("File saved successfully!")
                print("    File path:", os.path.join(os.path.dirname(self._excel_dirname), "neuromatch_res.xlsx"))
                print("    File path:", os.path.join(os.path.dirname(self._excel_dirname), "neuromatch_res.pkl"))
        except Exception as e:
            QMessageBox.critical(self, "File Save Error", f"An error occurred: {e}")
            
            
if __name__ == '__main__':
    # Run the application
    app = QApplication(sys.argv)
    mainWindow = NeuroMatchGUI()
    mainWindow.show()
    sys.exit(app.exec())
