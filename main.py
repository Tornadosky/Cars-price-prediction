import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import copy
warnings.filterwarnings("ignore")

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QLabel, QLineEdit,
 QCheckBox, QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QDockWidget, QListWidget, QVBoxLayout, QHBoxLayout,
 QTabWidget, QFrame, QPushButton, QCompleter)
from PyQt6.QtGui import QIcon, QAction, QPixmap
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

car_data = []
predictedPrice = 0

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

        self.loadData()

        self.initUI()
        self.show()

    def loadData(self):
        global car_data

        #for x in car_data.columns:
        #   car_data.drop(car_data.index[[y == '—' for y in car_data[x]]], inplace = True)
        #car_data.drop(car_data.index[[y < 10000 or y > 30000 for y in car_data["sellingprice"]]], inplace = True)
        #car_data = car_data.sample(frac=0.5).reset_index(drop=True)
    
        self.car_data = pd.read_csv('car_prices.csv', on_bad_lines='skip')
        self.car_data = self.car_data.dropna(how='any')
        self.car_data.drop(columns=['vin', 'seller', 'saledate','mmr'], inplace=True)

        # replace transmission with numbers
        self.car_data['transmission'].replace(['manual', 'automatic'],
                                [0, 1], inplace=True)
        # make every text occurance in lower-case
        for col in self.car_data.columns:
            if type(self.car_data[col][0]) is str:
                self.car_data[col] = self.car_data[col].apply(lambda x: x.lower())
        
        # assign to global variable
        car_data = self.car_data

        df_train=copy.deepcopy(car_data)

        cols=np.array(car_data.columns[car_data.dtypes != object])
        for i in df_train.columns:
            if i not in cols:
                df_train[i]=df_train[i].map(str)
        df_train.drop(columns=cols,inplace=True)

        # build dictionary function
        cols=np.array(car_data.columns[car_data.dtypes != object])
        d = defaultdict(LabelEncoder)

        # only for categorical columns apply dictionary by calling fit_transform 
        df_train = df_train.apply(lambda x: d[x.name].fit_transform(x))
        df_train[cols] = car_data[cols]

        ftrain = ['year', 'make', 'model', 'trim', 'body', 'transmission', 
                'state', 'condition', 'odometer', 'color', 'interior', 'sellingprice']

        def define_data():
            # define car dataset
            car_data2 = df_train[ftrain]
            X = car_data2.drop(columns=['sellingprice']).values
            y0 = car_data2['sellingprice'].values
            lab_enc = preprocessing.LabelEncoder()
            y = lab_enc.fit_transform(y0)
            return X, y
        
        # Create DecisionTreeRegressor model
        self.model = DecisionTreeRegressor()

        X, y = define_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
        self.model.fit(X_train,y_train)

        print(self.model.score(X_test, y_test))

    def showPrediction(self) :
        global predictedPrice
        
        predictedPrice = round(self.model.predict(___)[0], 2)
        print(type(predictedPrice))
        print("Predicted car price: %.2f" %predictedPrice)
        self.labPrediction.setText(f"Predicted car price: {predictedPrice}")

    def initUI(self):
        global car_data

        chart = Canvas(self)

        # Auto complete for QLineEdit()
        completer = QCompleter(car_data['make'].unique())
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        # Input field for Manufacturer with auto completion
        self.inputWid0Tab1 = QLineEdit()
        self.lblWid0Tab1 = QLabel("Manufacturer")
        self.inputWid0Tab1.setCompleter(completer)
        self.lblWid0Tab1.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Checkbox for Transmission (Auto - checked, Manual - unchecked)
        self.inputWid3Tab1 = QCheckBox("Transmission Auto")

        # Input field for Condition (from 1.0 to 5.0, step = 0.1)
        self.inputWid2Tab1 = QDoubleSpinBox()
        self.lblWid2Tab1 = QLabel("Condition")
        self.inputWid2Tab1.setDecimals(1)
        self.inputWid2Tab1.setRange(1.0, 5.0)
        self.inputWid2Tab1.setSingleStep(0.1)
        self.lblWid2Tab1.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Slider for Odometer (from 0 to max value in dataset)
        self.inputWid1Tab1 = QSlider(Qt.Orientation.Horizontal)
        self.lblWid1Tab1 = QLabel("Odometer")
        self.inputWid1Tab1.setMinimum(0)
        self.inputWid1Tab1.setMaximum(int(max(car_data['odometer'])))
        self.inputWid1Tab1.valueChanged.connect(self.updateOdometer)
        self.lblWid1Tab1.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        #inputLay3Tab1.addWidget(lblWid3Tab1)



        '''inputLay0Tab2 = QVBoxLayout()
        inputLay1Tab2 = QVBoxLayout()
        inputLay2Tab2 = QVBoxLayout()
        inputLay3Tab2 = QVBoxLayout()'''
        

        self.tabLay1, self.tabLay2, self.tabLay3 = QVBoxLayout(), QVBoxLayout(), QVBoxLayout()

        self.tabLay1.addWidget(self.lblWid0Tab1)
        self.tabLay1.addWidget(self.inputWid0Tab1)
        
        self.tabLay1.addWidget(self.lblWid1Tab1)
        self.tabLay1.addWidget(self.inputWid1Tab1)
        
        self.tabLay1.addWidget(self.lblWid2Tab1)
        self.tabLay1.addWidget(self.inputWid2Tab1)
        
        self.tabLay1.addWidget(self.inputWid3Tab1)

        self.tabLay1.addStretch()


        self.tab1, self.tab2, self.tab3 = QWidget(), QWidget(), QWidget()

        self.tab1.setLayout(self.tabLay1)
        self.tab2.setLayout(self.tabLay2)
        self.tab3.setLayout(self.tabLay3)


        self.tabWidget = QTabWidget()
        self.tabWidget.addTab(self.tab1, 'One')
        self.tabWidget.addTab(self.tab2, 'Two')
        self.tabWidget.addTab(self.tab3, 'Three')


        self.innerDockWidget = QWidget()

        self.outerTabWidLay = QVBoxLayout()
        self.outerTabWidLay.addWidget(self.tabWidget)
        self.innerDockWidget.setLayout(self.outerTabWidLay)


        self.dockWidget = QDockWidget("Quack")
        self.dockWidget.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.dockWidget.setWidget(self.innerDockWidget)
        # self.dockWidget.setStyleSheet("border-left: 1px solid grey; border-right: 1px solid grey;")


        self.lbl00 = QLabel('Manufacturer: ')
        self.lbl01 = QLabel('Transmission: ')
        self.lbl02 = QLabel('Condition: ')

        self.lbl10 = QLabel('Odometer: ')
        self.lbl11 = QLabel('State: ')
        self.lbl12 = QLabel('Color: ')

        self.lbl20 = QLabel('MMR: ')
        self.lbl21 = QLabel('Year: ')
        self.lbl22 = QLabel('Body: ')

        self.colLay0 = QVBoxLayout()
        self.colLay1 = QVBoxLayout()
        self.colLay2 = QVBoxLayout()
        self.colLay3 = QVBoxLayout()

        self.colLay0.addWidget(self.lbl00)
        self.colLay0.addWidget(self.lbl01)
        self.colLay0.addWidget(self.lbl02)

        self.colLay1.addWidget(self.lbl10)
        self.colLay1.addWidget(self.lbl11)
        self.colLay1.addWidget(self.lbl12)

        self.colLay2.addWidget(self.lbl20)
        self.colLay2.addWidget(self.lbl21)
        self.colLay2.addWidget(self.lbl22)

        self.colLay3.addWidget(QPushButton('Predict'))



        self.outerLblLay = QHBoxLayout()
        self.outerLblLay.addLayout(self.colLay0)
        self.outerLblLay.addLayout(self.colLay1)
        self.outerLblLay.addLayout(self.colLay2)
        self.outerLblLay.addLayout(self.colLay3)

        self.centWidget = QWidget()

        self.centLay = QVBoxLayout()
        self.centLay.addWidget(chart, stretch= 4)
        self.centLay.addLayout(self.outerLblLay, stretch= 1)
        self.centWidget.setLayout(self.centLay)
        #centWidget.setStyleSheet("border-top: 1px solid grey;") #background-color: white;")

        self.setCentralWidget(self.centWidget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dockWidget)



    
        exitAct = QAction(QIcon('exit.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(QApplication.instance().quit)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)
    
    def updateOdometer(self):
        val = self.inputWid1Tab1.value()
        self.lbl10.setText("Odometer: " + str(val))



def main():

    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
