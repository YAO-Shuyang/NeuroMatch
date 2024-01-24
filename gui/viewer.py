from PyQt6.QtWidgets import QMainWindow, QTableWidget, QTableWidgetItem
import numpy as np

class DataFrameViewer(QMainWindow):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DataFrame Viewer")
        self.setGeometry(100, 100, 600, 400)
        self.df = df
        self.initUI()

    def initUI(self):
        # Create a table widget
        self.tableWidget = QTableWidget()
        self.setCentralWidget(self.tableWidget)

        # Set up the table based on the DataFrame
        self.tableWidget.setColumnCount(len(self.df.columns))
        self.tableWidget.setRowCount(len(self.df.index))
        for j in range(self.df.shape[1]):
            self.tableWidget.setColumnWidth(j, 70)
        self.tableWidget.setHorizontalHeaderLabels(self.df.columns.astype(str))

        # Populate the table with data
        for i in range(self.df.shape[0]):
            for j in range(self.df.shape[1]):
                if np.isnan(self.df.iloc[i, j]):
                    item = QTableWidgetItem("")
                else:
                    item = QTableWidgetItem(str(int(self.df.iloc[i, j])))
                self.tableWidget.setItem(i, j, item)
