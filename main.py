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
 QCheckBox, QComboBox, QSlider, QSpinBox, QDockWidget, QListWidget, QVBoxLayout, QHBoxLayout,
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
        manufacturer_names = car_data['make'].unique()
        completer = QCompleter(manufacturer_names)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        inputWid0Tab1 = QLineEdit()
        lblWid0Tab1 = QLabel("Manufacturer")
        inputWid0Tab1.setCompleter(completer)
        lblWid0Tab1.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Checkbox for transmission (Auto - 1, Manual - 0)
        inputWid3Tab1 = QCheckBox("Transmission Auto")

        inputWid2Tab1 = QSpinBox()
        lblWid2Tab1 = QLabel("Condition")
        lblWid2Tab1.setAlignment(Qt.AlignmentFlag.AlignTop)

        inputWid1Tab1 = QSlider(Qt.Orientation.Horizontal)
        lblWid1Tab1 = QLabel("Odometer")
        lblWid1Tab1.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        #inputLay3Tab1.addWidget(lblWid3Tab1)



        '''inputLay0Tab2 = QVBoxLayout()
        inputLay1Tab2 = QVBoxLayout()
        inputLay2Tab2 = QVBoxLayout()
        inputLay3Tab2 = QVBoxLayout()'''
        

        tabLay1, tabLay2, tabLay3 = QVBoxLayout(), QVBoxLayout(), QVBoxLayout()

        tabLay1.addWidget(lblWid0Tab1)
        tabLay1.addWidget(inputWid0Tab1)
        
        tabLay1.addWidget(lblWid1Tab1)
        tabLay1.addWidget(inputWid1Tab1)
        
        tabLay1.addWidget(lblWid2Tab1)
        tabLay1.addWidget(inputWid2Tab1)
        
        tabLay1.addWidget(inputWid3Tab1)

        tabLay1.addStretch()


        tab1, tab2, tab3 = QWidget(), QWidget(), QWidget()

        tab1.setLayout(tabLay1)
        tab2.setLayout(tabLay2)
        tab3.setLayout(tabLay3)


        tabWidget = QTabWidget()
        tabWidget.addTab(tab1, 'One')
        tabWidget.addTab(tab2, 'Two')
        tabWidget.addTab(tab3, 'Three')


        innerDockWidget = QWidget()

        outerTabWidLay = QVBoxLayout()
        outerTabWidLay.addWidget(tabWidget)
        innerDockWidget.setLayout(outerTabWidLay)


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
