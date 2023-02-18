# Project description
Data-based decision system created with PyQt6 for price prediction of used cars.
For more detailed data description look at the project's [Wiki](https://github.com/Tornadosky/Cars-price-prediction/wiki).

# Installation
1. Clone the project
    ```
    git clone --depth 1 https://github.com/Tornadosky/Cars-price-prediction
    ```
2. Start a virtual environment in Python

    Create _venv_ for the project:
    ```
    cd parent_folder_for_project
    python -m venv project_name
    ```
    Activate the virtual environment:
    ```
    cd project_name
    Scripts\activate
    ```
3. Install packages 
    - `Prerequisites: see requirements.txt`
    
    or
    ```
    pip install -r requirements.txt
    ```

    If something wrong with modules, the problem can be with python interpreter (choose the one for venv).
4. Run _**data.py**_. 

    This will create a model and dump it into a _my_model.pkl_. The task should be completed within 5 minutes or less.

    Note: After running _**data.py**_ the user will see overview of the dataset in the terminal.

# Basic Usage
Run the _**main.py**_ file to start an application. 
Fill in all input areas in a _Dock_ widget in the _Tab 1_ on the right. You can see labels under the graph changing. Switch to _Tab 2_ and fill in the areas that are left. 
After choosing suitable conditions click _Predict_ button.
The predicted price will be shown at the bottom left of the app and in the terminal.

User can:
- exit the application (by pressing _Ctrl+Q_)
- save the current graph (by pressing _Ctrl+S_)
- open the histogram window
- resize windows

# Implementation of the Project
+ For the application free data source dataset ["Used Car Auction Prices"](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices) from _Kaggle_ was used.
+ The dataset is saved as _car_prices.csv_.
+ The data analysis with at least 3 statistical metrics over the input data was shown in _car_kaggle_sol.ipynb_.
+ After running the project the user will see a short data description (created with _Pandas_) in the _Terminal_. Moreover, there is a possibility to have a look at histograms in the application.
+ More than 3 input widgets were used (QLabel, QLineEdit, QSpinBox, QCheckBox etc.)
+ The data was trained with the RandomForestRegressor algorithm in _data.py_.
+ The model is saved in file with a _joblib_ module.
+ MatplotLib was used for data visualization.
+ Application reacts to the change of widgets value and responds with prediction and plot.
+ Data upload with a button was not added as it would slow down the work of the app.