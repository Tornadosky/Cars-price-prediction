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

filename = "my_model.pickle"

car_data = pd.read_csv('car_prices.csv', on_bad_lines='skip')
print("row number: ", len(car_data))
print("column number: ", len(car_data.columns))

car_data = car_data.dropna(how='any')
#for x in car_data.columns:
   #car_data.drop(car_data.index[[y == '—' for y in car_data[x]]], inplace = True)
#car_data.drop(car_data.index[[y < 10000 or y > 30000 for y in car_data["sellingprice"]]], inplace = True)
car_data.drop(columns=['vin', 'seller', 'saledate', 'mmr', 'trim'], inplace=True)
#car_data = car_data.sample(frac=0.5).reset_index(drop=True)

print("car_data shape")
print(car_data.shape)
print("car_data.head(8):")
print(car_data.head(8))
print("car_data.info():")
print(car_data.info())
print("car_data.describe():")
print(car_data.describe())
print("car_data.corr():")
print(car_data.corr())


car_data['transmission'].replace(['manual', 'automatic'],
                        [0, 1], inplace=True)

for col in car_data.columns:
    if type(car_data[col][0]) is str:
        car_data[col] = car_data[col].apply(lambda x: x.lower())

ftrain = ['year', 'make', 'model', 'body', 'transmission', 
          'state', 'condition', 'odometer', 'color', 'interior', 'sellingprice']
car_data = pd.DataFrame(car_data, columns=ftrain)

df_train=copy.deepcopy(car_data)

cols=np.array(car_data.columns[car_data.dtypes != object])
for i in df_train.columns:
    if i not in cols:
        df_train[i]=df_train[i].map(str)
df_train.drop(columns=cols,inplace=True)

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# build dictionary function
cols=np.array(car_data.columns[car_data.dtypes != object])
d = defaultdict(LabelEncoder)

# only for categorical columns apply dictionary by calling fit_transform 
df_train = df_train.apply(lambda x: d[x.name].fit_transform(x))
df_train[cols] = car_data[cols]

def define_data():
    # define car_dataset
    car_data2 = df_train[ftrain]
    X = car_data2.drop(columns=['sellingprice']).values
    y0 = car_data2['sellingprice'].values
    lab_enc = preprocessing.LabelEncoder()
    y = lab_enc.fit_transform(y0)
    return X, y

model = RandomForestRegressor()

X, y = define_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
model.fit(X_train,y_train)

print(model.score(X_test, y_test))

# save model
pickle.dump(model, open(filename, "wb"))
