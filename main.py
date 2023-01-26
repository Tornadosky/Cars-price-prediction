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
from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QLabel, QLineEdit, QSpinBox,
 QCheckBox, QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QDockWidget, QListWidget, QVBoxLayout, QHBoxLayout,
 QTabWidget, QFrame, QPushButton, QCompleter)
from PyQt6.QtGui import QIcon, QAction, QPixmap
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

car_make, car_model, car_year, car_color, car_interior = [0] * 5
car_odometer, car_body, car_condition, car_transmission, car_state = [0] * 5


car_data = []
predictedPrice = 0
xAxis = "condition"

class Canvas(FigureCanvas):
    def __init__(self, parent):
        global xAxis

        fig, self.ax = plt.subplots(figsize=(5, 4), dpi=80)
        super().__init__(fig)
        self.setParent(parent)

        """ 
        Matplotlib Script
        """
        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)
        
        self.ax.plot(t, s)

        self.ax.set(xlabel=xAxis, ylabel='voltage (mV)',
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
        self.car_data.drop(columns=['vin', 'seller', 'saledate','mmr', 'trim'], inplace=True)

        # replace transmission with numbers
        self.car_data['transmission'].replace(['manual', 'automatic'],
                                [0, 1], inplace=True)
        
        # make every text occurance in lower-case
        for col in self.car_data.columns:
            if type(self.car_data[col][0]) is str:
                self.car_data[col] = self.car_data[col].apply(lambda x: x.lower())
        
        # assign to global variable
        car_data = self.car_data

        df_train = copy.deepcopy(car_data)

        cols = np.array(car_data.columns[car_data.dtypes != object])
        for i in df_train.columns:
            if i not in cols:
                df_train[i] = df_train[i].map(str)
        df_train.drop(columns=cols, inplace=True)

        # build dictionary function
        cols = np.array(car_data.columns[car_data.dtypes != object])
        self.d = defaultdict(LabelEncoder)

        # only for categorical columns apply dictionary by calling fit_transform 
        df_train = df_train.apply(lambda x: self.d[x.name].fit_transform(x))
        
        df_train[cols] = car_data[cols]

        ftrain = ['year', 'make', 'model', 'body', 'transmission', 
                'state', 'condition', 'odometer', 'color', 'interior', 'sellingprice']

        self.lab_enc = preprocessing.LabelEncoder()

        def define_data():
            # define car dataset
            car_data2 = df_train[ftrain]
            X = car_data2.drop(columns=['sellingprice']).values
            y0 = car_data2['sellingprice'].values
            y = self.lab_enc.fit_transform(y0)
            return X, y
        
        # Create DecisionTreeRegressor model
        self.model = DecisionTreeRegressor()

        X, y = define_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
        self.model.fit(X_train,y_train)

        print(self.model.score(X_test, y_test))

    def showPrediction(self) :
        global car_make, car_model, car_year, car_color, car_interior
        global car_odometer, car_body, car_condition, car_transmission, car_state
        global predictedPrice, car_data

        ftrain = ['year', 'make', 'model', 'body', 'transmission', 
                'state', 'condition', 'odometer', 'color', 'interior']

        new_row = {'year':[car_year], 'make':[car_make], 'model':[car_model], 'body':[car_body],
         'transmission':[car_transmission], 'state':[car_state], 'condition':[car_condition], 'odometer':[car_odometer],
         'color':[car_color], 'interior':[car_interior]}
        df = pd.DataFrame.from_dict(new_row)
        print(df)
        df3 = copy.deepcopy(df)

        cols = np.array(df.columns[df.dtypes != object])
        for i in df.columns:
            if i not in cols:
                df3[i] = df3[i].map(str)
        df3.drop(columns=cols, inplace=True)

        print(df3)

        # only for categorical columns apply dictionary by calling fit_transform 
        df3 = df3.apply(lambda x: self.d[x.name].transform(x))
        df3[cols] = df[cols]

        my_car = df3[ftrain]

        print(my_car)
                
        predictedPrice = self.lab_enc.inverse_transform([int(self.model.predict(my_car)[0])])[0]
        print(f"Predicted car price: {predictedPrice} $")

    def initUI(self):
        global car_data
        global xAxis

        # Auto complete for QLineEdit()
        self.unique_manuf = car_data['make'].unique()
        completer = QCompleter(self.unique_manuf)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        # Input field for Manufacturer with auto completion
        self.inputMake = QLineEdit()
        self.tabLblMake = QLabel("Manufacturer")
        self.inputMake.setCompleter(completer)
        self.inputMake.textChanged.connect(self.updateMake)
        #self.tabLblMake.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Combobox for model depending on manufacturer
        self.inputModel = QComboBox()
        self.tabLblModel = QLabel("Model")
        self.inputModel.setPlaceholderText("Select Model...")
        self.inputModel.setEnabled(False)
        self.inputModel.currentTextChanged.connect(self.updateModel)

        # Combobox for body 
        self.inputBody = QComboBox()
        self.tabLblBody = QLabel("Body")
        self.inputBody.setPlaceholderText("Select Body...")
        self.inputBody.addItems(self.car_data['body'].unique())
        self.inputBody.currentTextChanged.connect(self.updateBody)

        # Checkbox for Transmission (Auto - checked, Manual - unchecked)
        self.inputTransmission = QCheckBox("Transmission Auto")
        self.inputTransmission.stateChanged.connect(self.updateTransmission)

        # Input field for Condition (from 1.0 to 5.0, step = 0.1)
        self.inputCondition = QDoubleSpinBox()
        self.tabLblCondition = QLabel("Condition")
        self.inputCondition.setDecimals(1)
        self.inputCondition.setRange(1.0, 5.0)
        self.inputCondition.setSingleStep(0.1)
        self.inputCondition.setValue(2.5)
        self.inputCondition.valueChanged.connect(self.updateCondition)

        # Slider for Odometer (from 0 to max value in dataset)
        max_odometer = int(max(car_data['odometer']))
        self.inputOdometer = QSlider(Qt.Orientation.Horizontal)
        self.tabLblOdometer = QLabel("Odometer")
        self.inputOdometer.setMinimum(0)
        self.inputOdometer.setMaximum(max_odometer)
        self.inputOdometer.valueChanged.connect(self.updateOdometer)

        # Combobox for choosing x-axis on the plot
        self.inputXaxis = QComboBox()
        self.lblxComboBox = QLabel("X-axis feature:")
        self.inputXaxis.setPlaceholderText("Select X-axis...")
        self.inputXaxis.addItems(self.car_data.columns)
        self.inputXaxis.currentTextChanged.connect(self.updateX)

        # Combobox for state
        self.inputState = QComboBox()
        self.tabLblState = QLabel("State")
        self.inputState.setPlaceholderText("Select State...")
        self.inputState.addItems(self.car_data['state'].unique())
        self.inputState.currentTextChanged.connect(self.updateState)

        # Combobox for color
        self.inputColor = QComboBox()
        self.tabLblColor = QLabel("Color")
        self.inputColor.setPlaceholderText("Select Color...")
        self.inputColor.addItems(self.car_data['color'].unique())
        self.inputColor.currentTextChanged.connect(self.updateColor)

        # Combobox for color
        self.inputInterior = QComboBox()
        self.tabLblInterior = QLabel("Interior")  
        self.inputInterior.setPlaceholderText("Select Interior...")
        self.inputInterior.addItems(self.car_data['interior'].unique())
        self.inputInterior.currentTextChanged.connect(self.updateInterior)

        # Input field for Year
        min_year = min(car_data['year'])
        max_year = max(car_data['year'])
        self.inputYear = QSpinBox()
        self.tabLblYear = QLabel("Year")
        self.inputYear.setMinimum(min_year)
        self.inputYear.setMaximum(max_year)
        self.inputYear.valueChanged.connect(self.updateYear)
        



        '''inputLay0Tab2 = QVBoxLayout()
        inputLay1Tab2 = QVBoxLayout()
        inputLay2Tab2 = QVBoxLayout()
        inputLay3Tab2 = QVBoxLayout()'''
        

        self.tabLay1, self.tabLay2 = QVBoxLayout(), QVBoxLayout()

        # Tab 1 widgets
        self.tabLay1.addWidget(self.tabLblMake)
        self.tabLay1.addWidget(self.inputMake)
        self.tabLay1.addSpacing(15)

        self.tabLay1.addWidget(self.tabLblModel)
        self.tabLay1.addWidget(self.inputModel)
        self.tabLay1.addSpacing(15)

        self.tabLay1.addWidget(self.tabLblBody)
        self.tabLay1.addWidget(self.inputBody)
        self.tabLay1.addSpacing(15)
        
        self.tabLay1.addWidget(self.tabLblOdometer)
        self.tabLay1.addWidget(self.inputOdometer)
        self.tabLay1.addSpacing(15)
        
        self.tabLay1.addWidget(self.inputTransmission)
        self.tabLay1.addSpacing(20)

        self.tabLay1.addWidget(self.lblxComboBox)
        self.tabLay1.addWidget(self.inputXaxis)

        self.tabLay1.addStretch()

        
        # Tab 2 widgets
        self.tabLay2.addWidget(self.tabLblState)
        self.tabLay2.addWidget(self.inputState)
        self.tabLay2.addSpacing(15)

        self.tabLay2.addWidget(self.tabLblColor)
        self.tabLay2.addWidget(self.inputColor)
        self.tabLay2.addSpacing(15)

        self.tabLay2.addWidget(self.tabLblInterior)
        self.tabLay2.addWidget(self.inputInterior)
        self.tabLay2.addSpacing(15)

        self.tabLay2.addWidget(self.tabLblCondition)
        self.tabLay2.addWidget(self.inputCondition)
        self.tabLay2.addSpacing(15)

        self.tabLay2.addWidget(self.tabLblYear)
        self.tabLay2.addWidget(self.inputYear)
        self.tabLay2.addSpacing(15)


        self.tab1, self.tab2 = QWidget(), QWidget()

        self.tab1.setLayout(self.tabLay1)
        self.tab2.setLayout(self.tabLay2)


        self.tabWidget = QTabWidget()
        self.tabWidget.addTab(self.tab1, 'One')
        self.tabWidget.addTab(self.tab2, 'Two')


        self.innerDockWidget = QWidget()

        self.outerTabWidLay = QVBoxLayout()
        self.outerTabWidLay.addWidget(self.tabWidget)
        self.innerDockWidget.setLayout(self.outerTabWidLay)


        self.dockWidget = QDockWidget("Dock")
        self.dockWidget.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.dockWidget.setWidget(self.innerDockWidget)
        # self.dockWidget.setStyleSheet("border-left: 1px solid grey; border-right: 1px solid grey;")


        # Information labels 
        self.lblMake = QLabel('Manufacturer: -')
        self.lblModel = QLabel('Model: -')
        self.lblBody = QLabel('Body: -')

        self.lblOdometer = QLabel('Odometer: -')
        self.lblYear = QLabel('Year: -')
        self.lblCondition = QLabel('Condition: -')

        self.lblState = QLabel('State: -')
        self.lblColor = QLabel('Color: -')
        self.lblInterior = QLabel('Interior: -')
        
        self.lblTransmission = QLabel('Transmission: Manual')
        

        self.colLay0 = QVBoxLayout()
        self.colLay1 = QVBoxLayout()
        self.colLay2 = QVBoxLayout()
        self.colLay3 = QVBoxLayout()

        # 1st column
        self.colLay0.addWidget(self.lblMake)
        self.colLay0.addWidget(self.lblModel)
        self.colLay0.addWidget(self.lblBody)

        # 2nd column
        self.colLay1.addWidget(self.lblState)
        self.colLay1.addWidget(self.lblColor)
        self.colLay1.addWidget(self.lblInterior)

        # 3rd column
        self.colLay2.addWidget(self.lblOdometer)
        self.colLay2.addWidget(self.lblYear)
        self.colLay2.addWidget(self.lblCondition)

        
        # Predict push-button
        self.btn = QPushButton("Predict")
        self.btn.setMinimumHeight(int(self.btn.height() / 10))
        self.btn.clicked.connect(self.showPrediction)
        
        # 4th column
        self.colLay3.addSpacing(3)
        self.colLay3.addWidget(self.lblTransmission)
        self.colLay3.addWidget(QWidget())
        self.colLay3.addWidget(self.btn)
        

        self.outerLblLay = QHBoxLayout()
        self.outerLblLay.addLayout(self.colLay0)
        self.outerLblLay.addLayout(self.colLay1)
        self.outerLblLay.addLayout(self.colLay2)
        self.outerLblLay.addLayout(self.colLay3)

        self.centWidget = QWidget()

        self.centLay = QVBoxLayout()
        chart = Canvas(self)
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
        global car_odometer
        car_odometer = self.inputOdometer.value()
        self.lblOdometer.setText("Odometer: " + str(car_odometer))
    
    def updateCondition(self):
        global car_condition
        car_condition = round(self.inputCondition.value(), 1)
        self.lblCondition.setText("Condition: " + str(car_condition))

    def updateX(self):
        global xAxis
        xAxis = self.inputXaxis.currentText()
    
    def updateMake(self):
        global car_make
        val = self.inputMake.text().lower()
        if val not in self.unique_manuf:
            val = None
        if val:
            self.inputModel.clear()
            self.inputModel.addItems(self.car_data.loc[self.car_data['make'] == val]['model'].unique())
            self.inputModel.setEnabled(True)
            car_make = val
            self.lblMake.setText("Manufacturer: " + val.capitalize())
        else:
            self.inputModel.setEnabled(False)
        
    def updateModel(self):
        global car_model
        car_model = self.inputModel.currentText()
        self.lblModel.setText("Model: " + car_model)

    def updateTransmission(self):
        global car_transmission

        val = self.inputTransmission.isChecked()
        if val:
            val = "Auto"
            car_transmission = 1
        else:
            val = "Manual"
            car_transmission = 0
        self.lblTransmission.setText("Transmission: " + val)

    def updateBody(self):
        global car_body
        car_body = self.inputBody.currentText()
        self.lblBody.setText("Body: " + car_body)

    def updateState(self):
        global car_state
        car_state = self.inputState.currentText()
        self.lblState.setText("State: " + car_state)

    def updateColor(self):
        global car_color
        car_color = self.inputColor.currentText()
        self.lblColor.setText("Color: " + car_color)
    
    def updateInterior(self):
        global car_interior
        car_interior = self.inputInterior.currentText()
        self.lblInterior.setText("Interior: " + car_interior)
    
    def updateYear(self):
        global car_year
        car_year = self.inputYear.value()
        self.lblYear.setText("Year: " + str(car_year))

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
