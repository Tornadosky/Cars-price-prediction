import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import copy
import joblib # for loading the model
warnings.filterwarnings("ignore")

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QLabel, QLineEdit, QSpinBox,
 QCheckBox, QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QDockWidget, QVBoxLayout, QHBoxLayout,
 QTabWidget, QPushButton, QCompleter)
from PyQt6 import QtGui
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# initial global variables

# car dict for user's input 
car = { 'year': None, 'make': None, 'model': None, 'body': None,
         'transmission': 0, 'state': None, 'condition': None, 'odometer': None,
         'color': None, 'interior': None }
car_data = []
predictedPrice = 0
# xAxis is a chosen by user independent feature to be displayed on plot
xAxis = 'odometer'

# path for saving images
IMAGES_PATH = Path() / "images" 
IMAGES_PATH.mkdir(parents=True, exist_ok=True)
def saveFig(fig, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        fig.tight_layout()
    fig.savefig(path, format=fig_extension, dpi=resolution)


class Canvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=80)
        super().__init__(self.fig)

class SecondWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        global car_data

        self.setWindowTitle('Histograms')
        self.setGeometry(310, 110, 750, 450)
        
        secWindLay = QHBoxLayout()
        secWindWid = QWidget()

        self.chart2 = Canvas(self)
        car_data.hist(ax=self.chart2.ax, bins=50, figsize=(20,15))

        secWindLay.addWidget(self.chart2)
        secWindWid.setLayout(secWindLay)
        self.setCentralWidget(secWindWid)
        
        saveFig(self.chart2.fig,"histogram", tight_layout=True, fig_extension="png", resolution=300)

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
        
        X, y = define_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

        # load model
        self.model = joblib.load('my_model.pkl')

        print(self.model.score(X_test, y_test))

    def showPrediction(self) :
        global xAxis, predictedPrice, car_data, car

        ftrain = ['year', 'make', 'model', 'body', 'transmission', 
                'state', 'condition', 'odometer', 'color', 'interior']
                
        new_row = { 'year':[car['year']], 'make':[car['make']], 'model':[car['model']], 'body':[car['body']],
         'transmission':[car['transmission']], 'state':[car['state']], 'condition':[car['condition']], 'odometer':[car['odometer']],
         'color':[car['color']], 'interior':[car['interior']] }

        new_row_vals = [i[0] for i in list(new_row.values())]

        # return Invalid input in case of none values
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
        
        # clear the chart
        self.chart.ax.cla()

        # get x, y values for plot
        if xAxis == 'model':
            plot_df = car_data.loc[car_data['make'] == car['make']].loc[:, ('sellingprice', xAxis)]
        else:
            plot_df = car_data.loc[:, ('sellingprice', xAxis)]
        
        plot_df.sort_values(by=[xAxis], ascending=True, inplace=True)

        # appropriate range for different features
        # e.g. for odometer not to have too many unique vals
        if isinstance(car_data[xAxis][0], str) or xAxis != "odometer":
            x_range = plot_df[xAxis].unique()
        else:
            chunk_size = 1000
            min_val, max_val = min(plot_df[xAxis]), max(plot_df[xAxis])
            interval_step = (max_val - min_val) // chunk_size

            x_range = plot_df[xAxis].unique()[int(min_val)::int(interval_step)]
        
        # x - feature (values of xAxis)
        # y - average price of specific x
        x_vals = []
        y_vals = []
        
        # get average price for each x
        for elem in x_range:
            price_list = plot_df.loc[plot_df[xAxis] == elem]['sellingprice'].values

            if len(price_list):
                x_vals.append(elem)
                y_vals.append(sum(price_list) // len(price_list))
            
        # make dictionary, keys will become dataframe column names
        intermediate_dictionary = {'sellingprice':y_vals, xAxis:x_vals}

        # convert dictionary to Pandas dataframe
        plot_df = pd.DataFrame(intermediate_dictionary)
        plot_df.reset_index(drop = True, inplace = True)

        # if x values are str plot bars, else dots
        if isinstance(car_data[xAxis][0], str):
            plot_df.set_index(xAxis).plot(ax=self.chart.ax, kind='bar')
        else:
            plot_df.set_index(xAxis).plot(ax=self.chart.ax, marker=".", markersize=3)
        
        # plot our predicted price with marker
        if car[xAxis] and predictedPrice:
            # check if car[xAxis] is str, if yes find index otherwise
            # marker is to the left for xAxises which are str
            x_value = car[xAxis]
            if isinstance(x_value, str):
                x_value = x_vals.index(x_value)
            # plot the marker
            self.chart.ax.plot(x_value, predictedPrice, marker="^", linestyle="", alpha=0.8, c='red')
      
        self.chart.fig.tight_layout()
        self.chart.draw()
        
    def showSecondWindow(self):
        if self.win2.isHidden(): 
            self.win2.show()

    def initUI(self):
        global car_data
        global xAxis, predictedPrice

        self.win2 = SecondWindow()

        QApplication.setWindowIcon(QtGui.QIcon('logo.png'))

        # auto complete for QLineEdit()
        self.unique_manuf = car_data['make'].unique()
        completer = QCompleter(self.unique_manuf)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        # input field for Manufacturer with auto completion
        self.inputMake = QLineEdit()
        tabLblMake = QLabel("Manufacturer")
        self.inputMake.setCompleter(completer)
        self.inputMake.textChanged.connect(self.updateMake)

        # combobox for model depending on manufacturer
        self.inputModel = QComboBox()
        tabLblModel = QLabel("Model")
        self.inputModel.setPlaceholderText("Select Model...")
        self.inputModel.setEnabled(False)
        self.inputModel.currentTextChanged.connect(self.updateModel)

        # combobox for body 
        self.inputBody = QComboBox()
        tabLblBody = QLabel("Body")
        self.inputBody.setPlaceholderText("Select Body...")
        self.inputBody.addItems(np.sort(car_data['body'].unique()))
        self.inputBody.currentTextChanged.connect(self.updateBody)

        # checkbox for Transmission (Auto - checked, Manual - unchecked)
        self.inputTransmission = QCheckBox("Transmission Auto")
        self.inputTransmission.stateChanged.connect(self.updateTransmission)

        # input field for Condition (from 1.0 to 5.0, step = 0.1)
        self.inputCondition = QDoubleSpinBox()
        tabLblCondition = QLabel("Condition")
        self.inputCondition.setDecimals(1)
        self.inputCondition.setRange(1.0, 5.0)
        self.inputCondition.setSingleStep(0.1)
        self.inputCondition.setValue(2.5)
        self.inputCondition.valueChanged.connect(self.updateCondition)

        # slider for Odometer (from 0 to max value in dataset)
        max_odometer = int(max(car_data['odometer']))
        self.inputOdometer = QSlider(Qt.Orientation.Horizontal)
        tabLblOdometer = QLabel("Odometer")
        self.inputOdometer.setMinimum(0)
        self.inputOdometer.setMaximum(max_odometer // 3)
        self.inputOdometer.valueChanged.connect(self.updateOdometer)

        # combobox for choosing x-axis on the plot
        self.inputXaxis = QComboBox()
        lblxComboBox = QLabel("X-axis feature:")
        self.inputXaxis.setPlaceholderText("Select X-axis...")
        self.inputXaxis.addItems(np.sort(car_data.columns.drop('sellingprice')))
        self.inputXaxis.currentTextChanged.connect(self.updateX)

        # combobox for state
        self.inputState = QComboBox()
        tabLblState = QLabel("US State")
        self.inputState.setPlaceholderText("Select US State...")
        self.inputState.addItems(np.sort(car_data['state'].unique()))
        self.inputState.currentTextChanged.connect(self.updateState)

        # combobox for color
        self.inputColor = QComboBox()
        tabLblColor = QLabel("Color")
        self.inputColor.setPlaceholderText("Select Color...")
        self.inputColor.addItems(np.sort(car_data['color'].unique()))
        self.inputColor.currentTextChanged.connect(self.updateColor)

        # combobox for interior color
        self.inputInterior = QComboBox()
        tabLblInterior = QLabel("Interior")  
        self.inputInterior.setPlaceholderText("Select Interior...")
        self.inputInterior.addItems(np.sort(car_data['interior'].unique()))
        self.inputInterior.currentTextChanged.connect(self.updateInterior)

        # input field for Year
        min_year = min(car_data['year'])
        max_year = max(car_data['year'])
        self.inputYear = QSpinBox()
        tabLblYear = QLabel("Year")
        self.inputYear.setMinimum(min_year)
        self.inputYear.setMaximum(max_year)
        self.inputYear.setValue(2008)
        self.inputYear.valueChanged.connect(self.updateYear)
        

        tabLay1, tabLay2 = QVBoxLayout(), QVBoxLayout()

        # tab 1 widgets
        tabLay1.addWidget(tabLblMake)
        tabLay1.addWidget(self.inputMake)
        tabLay1.addSpacing(15)

        tabLay1.addWidget(tabLblModel)
        tabLay1.addWidget(self.inputModel)
        tabLay1.addSpacing(15)

        tabLay1.addWidget(tabLblBody)
        tabLay1.addWidget(self.inputBody)
        tabLay1.addSpacing(15)
        
        tabLay1.addWidget(tabLblOdometer)
        tabLay1.addWidget(self.inputOdometer)
        tabLay1.addSpacing(15)
        
        tabLay1.addWidget(self.inputTransmission)
        tabLay1.addSpacing(20)

        tabLay1.addWidget(lblxComboBox)
        tabLay1.addWidget(self.inputXaxis)

        tabLay1.addStretch()

        
        # tab 2 widgets
        tabLay2.addWidget(tabLblState)
        tabLay2.addWidget(self.inputState)
        tabLay2.addSpacing(15)

        tabLay2.addWidget(tabLblColor)
        tabLay2.addWidget(self.inputColor)
        tabLay2.addSpacing(15)

        tabLay2.addWidget(tabLblInterior)
        tabLay2.addWidget(self.inputInterior)
        tabLay2.addSpacing(15)

        tabLay2.addWidget(tabLblCondition)
        tabLay2.addWidget(self.inputCondition)
        tabLay2.addSpacing(15)

        tabLay2.addWidget(tabLblYear)
        tabLay2.addWidget(self.inputYear)
        tabLay2.addSpacing(15)



        # tabs for TabWidget
        tab1, tab2 = QWidget(), QWidget()

        tab1.setLayout(tabLay1)
        tab2.setLayout(tabLay2)


        tabWidget = QTabWidget()
        tabWidget.addTab(tab1, 'Tab 1')
        tabWidget.addTab(tab2, 'Tab 2')


        innerDockWidget = QWidget()

        outerTabWidLay = QVBoxLayout()
        outerTabWidLay.addWidget(tabWidget)
        innerDockWidget.setLayout(outerTabWidLay)

        # dockWidget
        dockWidget = QDockWidget("Dock")
        dockWidget.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dockWidget.setWidget(innerDockWidget)


        # information labels 
        self.lblMake = QLabel('Manufacturer: -')
        self.lblModel = QLabel('Model: -')
        self.lblBody = QLabel('Body: -')

        self.lblOdometer = QLabel('Odometer: -')
        self.lblYear = QLabel('Year: -')
        self.lblCondition = QLabel('Condition: -')

        self.lblState = QLabel('US State: -')
        self.lblColor = QLabel('Color: -')
        self.lblInterior = QLabel('Interior: -')
        
        self.lblTransmission = QLabel('Transmission: Manual')
        

        colLay0 = QVBoxLayout()
        colLay1 = QVBoxLayout()
        colLay2 = QVBoxLayout()
        colLay3 = QVBoxLayout()

        # 1st column
        colLay0.addWidget(self.lblMake)
        colLay0.addWidget(self.lblModel)
        colLay0.addWidget(self.lblBody)

        # 2nd column
        colLay1.addWidget(self.lblState)
        colLay1.addWidget(self.lblColor)
        colLay1.addWidget(self.lblInterior)

        # 3rd column
        colLay2.addWidget(self.lblOdometer)
        colLay2.addWidget(self.lblYear)
        colLay2.addWidget(self.lblCondition)

        
        # predict push-button
        btn = QPushButton("Predict")
        btn.setMinimumHeight(int(btn.height() / 10))
        btn.clicked.connect(self.showPrediction)
        
        # 4th column
        colLay3.addSpacing(3)
        colLay3.addWidget(self.lblTransmission)
        colLay3.addWidget(QWidget())
        colLay3.addWidget(btn)
        

        outerLblLay = QHBoxLayout()
        outerLblLay.addLayout(colLay0)
        outerLblLay.addLayout(colLay1)
        outerLblLay.addLayout(colLay2)
        outerLblLay.addLayout(colLay3)

        centWidget = QWidget()

        centLay = QVBoxLayout()
        self.chart = Canvas(self)

        self.plotPrice()

        centLay.addWidget(self.chart, stretch= 4)
        centLay.addLayout(outerLblLay, stretch= 1)
        centWidget.setLayout(centLay)

        self.setCentralWidget(centWidget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dockWidget)

        # exit application action
        exitAct = QAction(QIcon('exit.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(QApplication.instance().quit)

        # save plot action
        savePlt = QAction(QIcon('save.png'), '&Save', self)
        savePlt.setShortcut('Ctrl+S')
        savePlt.setStatusTip('Save Plot')
        savePlt.triggered.connect(self.savePlot)

        # Show histograms in win2 action
        showHist = QAction(QIcon("graph.png"), "&Histogram", self)
        showHist.setStatusTip("Show histograms")
        showHist.triggered.connect(self.showSecondWindow)

        # Show histograms in win2 action
        showHist = QAction(QIcon("graph.png"), "&Histogram", self)
        showHist.setStatusTip("Show histograms")
        showHist.triggered.connect(self.showSecondWindow)

        # update status bar on change
        self.statusBar().messageChanged.connect(self.updateStatus)
        self.statusBar().hide()

        # menubar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)
        fileMenu.addAction(savePlt)
        fileMenu.addAction(showHist)
    
    # save figures to images/ with unique names
    def savePlot(self):
        global xAxis, predictedPrice
        saveFig(self.chart.fig, f"prediction_plot_{xAxis}_{predictedPrice}", tight_layout=True, fig_extension="png", resolution=300)

    # update status bar
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
    
    # update manufacturer
    # when chosen enable ComboBox for models
    def updateMake(self):
        global car, car_data
        val = self.inputMake.text().lower()
        # if wrong input ignore it
        if val not in self.unique_manuf:
            val = None

        if val:
            self.inputModel.clear()
            # find all models for specific manufacturer
            self.inputModel.addItems(np.sort(car_data.loc[car_data['make'] == val]['model'].unique()))
            self.inputModel.setEnabled(True)
            car['make'] = val

            if val == 'bmw':
                self.lblMake.setText("Manufacturer: " + val.upper())
            else:
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
        self.lblState.setText("US State: " + car['state'].upper())

    def updateColor(self):
        global car
        car['color'] = self.inputColor.currentText()
        self.lblColor.setText("Color: " + car['color'].capitalize())
    
    def updateInterior(self):
        global car
        car['interior'] = self.inputInterior.currentText()
        self.lblInterior.setText("Interior: " + car['interior'].capitalize())
    
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
