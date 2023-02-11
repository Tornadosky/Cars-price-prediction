import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import copy
import pickle
warnings.filterwarnings("ignore")

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QLabel, QLineEdit, QSpinBox,
 QCheckBox, QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QDockWidget, QListWidget, QVBoxLayout, QHBoxLayout,
 QTabWidget, QFrame, QPushButton, QCompleter, QStatusBar)
from PyQt6.QtGui import QIcon, QAction, QPixmap
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Initial global variables
# Car dict for user's input 
car = { 'year': None, 'make': None, 'model': None, 'body': None,
         'transmission': 0, 'state': None, 'condition': None, 'odometer': None,
         'color': None, 'interior': None }
car_data = []
predictedPrice = 0
# File for model
filename = "my_model.pickle"
# xAxis is a chosen by user independent feature to be displayed on plot
xAxis = 'odometer'

IMAGES_PATH = Path() / "images" 
IMAGES_PATH.mkdir(parents=True, exist_ok=True)
def save_fig(fig, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        fig.tight_layout()
    fig.savefig(path, format=fig_extension, dpi=resolution)


class Canvas(FigureCanvas):
    def __init__(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=80)
        super().__init__(self.fig)
        self.setParent(parent)


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
        self.car_data = pd.read_csv('car_prices.csv', on_bad_lines='skip')
        self.car_data = self.car_data.dropna(how='any')
        self.car_data.drop(columns=['vin', 'seller', 'saledate','mmr', 'trim'], inplace=True)

        # assign to global variable
        car_data = self.car_data

        # load model
        self.model = pickle.load(open(filename, "rb"))

    def showPrediction(self) :
        global xAxis, predictedPrice, car_data, car

        ftrain = ['year', 'make', 'model', 'body', 'transmission', 
                'state', 'condition', 'odometer', 'color', 'interior']
                
        new_row = { 'year':[car['year']], 'make':[car['make']], 'model':[car['model']], 'body':[car['body']],
         'transmission':[car['transmission']], 'state':[car['state']], 'condition':[car['condition']], 'odometer':[car['odometer']],
         'color':[car['color']], 'interior':[car['interior']] }

        new_row_vals = [i[0] for i in list(new_row.values())]

        if None in new_row_vals:
            self.statusBar().setStyleSheet("background-color : red")
            self.statusBar().showMessage("Invalid Input!")
            print("Invalid Input!")
            return 

        df = pd.DataFrame.from_dict(new_row)

        print(df)
        df3 = copy.deepcopy(df)

        cols = np.array(df.columns[df.dtypes != object])
        for i in df.columns:
            if i not in cols:
                df3[i] = df3[i].map(str)
        df3.drop(columns=cols, inplace=True)

        # only for categorical columns apply dictionary by calling fit_transform 
        df3 = df3.apply(lambda x: self.d[x.name].transform(x))
        df3[cols] = df[cols]

        my_car = df3[ftrain]
        
        transformed_pred = [int(self.model.predict(my_car)[0])]    
        predictedPrice = self.lab_enc.inverse_transform(transformed_pred)[0]
        self.statusBar().showMessage(f"Predicted car price: {predictedPrice} $")
        print(f"Predicted car price: {predictedPrice} $")

        self.plotPrice()

    def plotPrice(self):
        global xAxis, predictedPrice, car_data, car
        
        # Clear the chart
        self.chart.ax.cla()

        # Get x, y values for plot
        if xAxis == 'model':
            plot_df = car_data.loc[car_data['make'] == car['make']].loc[:, ('sellingprice', xAxis)]
        else:
            plot_df = car_data.loc[:, ('sellingprice', xAxis)]
        
        plot_df.sort_values(by=[xAxis], ascending=True, inplace=True)

        # Appropriate range for different features
        if isinstance(car_data[xAxis][0], str) or xAxis != "odometer":
            x_range = plot_df[xAxis].unique()
        else:
            chunk_size = 1000
            min_val, max_val = min(plot_df[xAxis]), max(plot_df[xAxis])
            interval_step = (max_val - min_val) // chunk_size

            x_range = plot_df[xAxis].unique()[int(min_val)::int(interval_step)]

        # Get average price for each x
        x_vals = []
        y_vals = []
        
        for elem in x_range:
            price_list = plot_df.loc[plot_df[xAxis] == elem]['sellingprice'].values

            if len(price_list):
                x_vals.append(elem)
                y_vals.append(sum(price_list) // len(price_list))
            
        # Make dictionary, keys will become dataframe column names
        intermediate_dictionary = {'sellingprice':y_vals, xAxis:x_vals}

        # Convert dictionary to Pandas dataframe
        plot_df = pd.DataFrame(intermediate_dictionary)
        plot_df.reset_index(drop = True, inplace = True)
        if isinstance(car_data[xAxis][0], str):
            plot_df.set_index(xAxis).plot(kind='bar', ax=self.chart.ax)
        else:
            plot_df.set_index(xAxis).plot(ax=self.chart.ax)

        # Plot our predicted price with marker
        if car[xAxis] and predictedPrice:
            self.chart.ax.plot(car[xAxis], predictedPrice, marker="^", linestyle="", alpha=0.8, c='red')
        
        self.chart.draw()
        
    def initUI(self):
        global car_data
        global xAxis, predictedPrice

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
        self.inputBody.addItems(car_data['body'].unique())
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
        self.inputOdometer.setMaximum(max_odometer // 3)
        self.inputOdometer.valueChanged.connect(self.updateOdometer)

        # Combobox for choosing x-axis on the plot
        self.inputXaxis = QComboBox()
        self.lblxComboBox = QLabel("X-axis feature:")
        self.inputXaxis.setPlaceholderText("Select X-axis...")
        self.inputXaxis.addItems(car_data.columns.drop('sellingprice')) 
        self.inputXaxis.currentTextChanged.connect(self.updateX)

        # Combobox for state
        self.inputState = QComboBox()
        self.tabLblState = QLabel("State")
        self.inputState.setPlaceholderText("Select State...")
        self.inputState.addItems(car_data['state'].unique())
        self.inputState.currentTextChanged.connect(self.updateState)

        # Combobox for color
        self.inputColor = QComboBox()
        self.tabLblColor = QLabel("Color")
        self.inputColor.setPlaceholderText("Select Color...")
        self.inputColor.addItems(car_data['color'].unique())
        self.inputColor.currentTextChanged.connect(self.updateColor)

        # Combobox for interior color
        self.inputInterior = QComboBox()
        self.tabLblInterior = QLabel("Interior")  
        self.inputInterior.setPlaceholderText("Select Interior...")
        self.inputInterior.addItems(car_data['interior'].unique())
        self.inputInterior.currentTextChanged.connect(self.updateInterior)

        # Input field for Year
        min_year = min(car_data['year'])
        max_year = max(car_data['year'])
        self.inputYear = QSpinBox()
        self.tabLblYear = QLabel("Year")
        self.inputYear.setMinimum(min_year)
        self.inputYear.setMaximum(max_year)
        self.inputYear.valueChanged.connect(self.updateYear)
        

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



        # Tabs for TabWidget
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

        # DockWidget
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
        self.chart = Canvas(self)

        self.plotPrice()

        self.centLay.addWidget(self.chart, stretch= 4)
        self.centLay.addLayout(self.outerLblLay, stretch= 1)
        self.centWidget.setLayout(self.centLay)
        #centWidget.setStyleSheet("border-top: 1px solid grey;") #background-color: white;")

        self.setCentralWidget(self.centWidget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dockWidget)

        # Exit application action
        exitAct = QAction(QIcon('exit.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(QApplication.instance().quit)

        # Save plot action
        savePlt = QAction(QIcon('save.png'), '&Save', self)
        savePlt.setShortcut('Ctrl+S')
        savePlt.setStatusTip('Save Plot')
        savePlt.triggered.connect(self.savePlot)

        #self.statusBar = QStatusBar()
        #self.setStatusBar(self.statusBar)
        self.statusBar().messageChanged.connect(self.updateStatus)
        self.statusBar().hide()

        # Menubar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)
        fileMenu.addAction(savePlt)
    
    def savePlot(self):
        global xAxis, predictedPrice
        save_fig(self.chart.figure, f"prediction_plot_{xAxis}_{predictedPrice}", tight_layout=True, fig_extension="png", resolution=300)

    def updateStatus(self):
        val = self.statusBar().currentMessage()
        if not val:
            self.statusBar().hide()
        elif val != "Invalid Input!":
            self.statusBar().setStyleSheet("")
            self.statusBar().show()
        else:
            self.statusBar().show()
    
    def updateOdometer(self):
        global car
        car['odometer'] = self.inputOdometer.value()
        self.lblOdometer.setText("Odometer: " + str(car['odometer']))
    
    def updateCondition(self):
        global car
        car['condition'] = round(self.inputCondition.value(), 1)
        self.lblCondition.setText("Condition: " + str(car['condition']))

    def updateX(self):
        global xAxis
        xAxis = self.inputXaxis.currentText()
    
    def updateMake(self):
        global car, car_data
        val = self.inputMake.text().lower()
        if val not in self.unique_manuf:
            val = None
        if val:
            self.inputModel.clear()
            self.inputModel.addItems(car_data.loc[car_data['make'] == val]['model'].unique())
            self.inputModel.setEnabled(True)
            car['make'] = val
            self.lblMake.setText("Manufacturer: " + val.capitalize())
            car['model'] = None
        else:
            self.inputModel.setEnabled(False)
            car['model'] = None
        
    def updateModel(self):
        global car
        car['model'] = self.inputModel.currentText()
        self.lblModel.setText("Model: " + car['model'])

    def updateTransmission(self):
        global car

        val = self.inputTransmission.isChecked()
        if val:
            val = "Auto"
            car['transmission'] = 1
        else:
            val = "Manual"
            car['transmission'] = 0
        self.lblTransmission.setText("Transmission: " + val)

    def updateBody(self):
        global car
        car['body'] = self.inputBody.currentText()
        self.lblBody.setText("Body: " + car['body'])

    def updateState(self):
        global car
        car['state'] = self.inputState.currentText()
        self.lblState.setText("State: " + car['state'])

    def updateColor(self):
        global car
        car['color'] = self.inputColor.currentText()
        self.lblColor.setText("Color: " + car['color'])
    
    def updateInterior(self):
        global car
        car['interior'] = self.inputInterior.currentText()
        self.lblInterior.setText("Interior: " + car['interior'])
    
    def updateYear(self):
        global car
        car['year'] = self.inputYear.value()
        self.lblYear.setText("Year: " + str(car['year']))

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
