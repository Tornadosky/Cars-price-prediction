import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QLabel, QLineEdit,
 QCheckBox, QComboBox, QSlider, QSpinBox, QDockWidget, QListWidget, QVBoxLayout, QHBoxLayout,
 QTabWidget, QFrame, QPushButton)
from PyQt6.QtGui import QIcon, QAction, QPixmap
from PyQt6.QtCore import Qt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class Canvas(FigureCanvas):
    def __init__(self, parent):
        fig, self.ax = plt.subplots(figsize=(5, 4), dpi=80)
        super().__init__(fig)
        self.setParent(parent)

        """ 
        Matplotlib Script
        """
        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)
        
        self.ax.plot(t, s)

        self.ax.set(xlabel='time (s)', ylabel='voltage (mV)',
               title='About as simple as it gets, folks')
        self.ax.grid()


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('Car price prediction')
        self.setGeometry(300, 100, 850, 500)

        self.initUI()
        self.show()
    

    def initUI(self):

        chart = Canvas(self)

        tab1, tab2, tab3 = QLabel("Hello there)"), QWidget(), QWidget()
        tabWidget = QTabWidget()
        tabWidget.addTab(tab1, 'One')
        tabWidget.addTab(tab2, 'Two')
        tabWidget.addTab(tab3, 'Three')


        innerDockWidget = QWidget()

        tabLay = QVBoxLayout()
        tabLay.addWidget(tabWidget)
        innerDockWidget.setLayout(tabLay)


        dockWidget = QDockWidget("Quack")
        dockWidget.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dockWidget.setWidget(innerDockWidget)
        #dockWidget.setStyleSheet("border-left: 1px solid grey; border-right: 1px solid grey;")


        lbl00 = QLabel('Manufacturer: ')
        lbl01 = QLabel('Transmission: ')
        lbl02 = QLabel('Condition: ')

        lbl10 = QLabel('Odometer: ')
        lbl11 = QLabel('State: ')
        lbl12 = QLabel('Color: ')

        lbl20 = QLabel('MMR: ')
        lbl21 = QLabel('Year: ')
        lbl22 = QLabel('Body: ')

        colLay0 = QVBoxLayout()
        colLay1 = QVBoxLayout()
        colLay2 = QVBoxLayout()
        colLay3 = QVBoxLayout()

        colLay0.addWidget(lbl00)
        colLay0.addWidget(lbl01)
        colLay0.addWidget(lbl02)

        colLay1.addWidget(lbl10)
        colLay1.addWidget(lbl11)
        colLay1.addWidget(lbl12)

        colLay2.addWidget(lbl20)
        colLay2.addWidget(lbl21)
        colLay2.addWidget(lbl22)

        colLay3.addWidget(QPushButton('Predict'))



        outerLblLay = QHBoxLayout()
        outerLblLay.addLayout(colLay0)
        outerLblLay.addLayout(colLay1)
        outerLblLay.addLayout(colLay2)
        outerLblLay.addLayout(colLay3)

        centWidget = QWidget()

        centLay = QVBoxLayout()
        centLay.addWidget(chart, stretch= 4)
        centLay.addLayout(outerLblLay, stretch= 1)
        centWidget.setLayout(centLay)
        #centWidget.setStyleSheet("border-top: 1px solid grey;") #background-color: white;")

        self.setCentralWidget(centWidget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dockWidget)



    
        exitAct = QAction(QIcon('exit.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(QApplication.instance().quit)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)




def main():

    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
