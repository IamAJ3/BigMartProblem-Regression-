#Importing libraries
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

df_train = pd.read_csv('Train_BigMart.csv')
df_train.head()

#Checking null values in each
print(df_train.isnull().sum())

df_train['Item_Fat_Content'].unique()
df_train['Item_Fat_Content'] = ['Low Fat' if x == 'LF' else 'Low Fat' if x == 'low fat' else 'Regular' if x == 'reg' else x for x in df_train['Item_Fat_Content']]

'''plt.figure(figsize=(25,15))
sns.countplot(x = 'Item_Type', data = df_train)'''

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_train['Outlet_Type'] = le.fit_transform(df_train['Outlet_Type'])


df_train.loc[ (pd.isnull(df_train['Outlet_Size'])) & (df_train['Outlet_Type'] == 0), 'Outlet_Size'] = 'Small'
df_train.loc[ (pd.isnull(df_train['Outlet_Size'])) & (df_train['Outlet_Type'] == 1), 'Outlet_Size'] = 'Small'
df_train.loc[ (pd.isnull(df_train['Outlet_Size'])) & (df_train['Outlet_Type'] == 2), 'Outlet_Size'] = 'Medium'
df_train.loc[ (pd.isnull(df_train['Outlet_Size'])) & (df_train['Outlet_Type'] == 3), 'Outlet_Size'] = 'Medium'


#Checking null values in each
print(df_train.isnull().sum())

df_train['Item_Weight'] = df_train['Item_Weight'].fillna(value = df_train['Item_Weight'].mean())
df_train.info()


le_list = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type']
for item in le_list:
    df_train[item] = le.fit_transform(df_train[item])


df_train.info()


len(df_train.columns)


train_list = np.arange(1,11)
X = df_train.iloc[:, train_list].values
X = X.astype(np.float32)

y = df_train.iloc[:, 11].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=100, booster = 'gbtree' , learning_rate=0.035, gamma=50, subsample=0.5,
                           colsample_bytree= 1, max_depth=3, tree_method = 'hist')
xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt


rms_xgb = sqrt(mean_squared_error(y_test, y_pred_xgb))
rms_xgb

df_test = pd.read_csv('Test_BigMart.csv')
df_test.head()

#Checking null values in each
print(df_test.isnull().sum())

df_test['Item_Fat_Content'] = ['Low Fat' if x == 'LF' else 'Low Fat' if x == 'low fat' else 'Regular' if x == 'reg' else x for x in df_test['Item_Fat_Content']]
df_test['Item_Fat_Content'].unique()

df_test['Outlet_Type'] = le.fit_transform(df_test['Outlet_Type'])

df_test.loc[ (pd.isnull(df_test['Outlet_Size'])) & (df_test['Outlet_Type'] == 0), 'Outlet_Size'] = 'Small'
df_test.loc[ (pd.isnull(df_test['Outlet_Size'])) & (df_test['Outlet_Type'] == 1), 'Outlet_Size'] = 'Small'
df_test.loc[ (pd.isnull(df_test['Outlet_Size'])) & (df_test['Outlet_Type'] == 2), 'Outlet_Size'] = 'Medium'
df_test.loc[ (pd.isnull(df_test['Outlet_Size'])) & (df_test['Outlet_Type'] == 3), 'Outlet_Size'] = 'Medium'

#Checking null values in each
print(df_test.isnull().sum())

df_test['Item_Weight'] = df_test['Item_Weight'].fillna(value = df_test['Item_Weight'].mean())
#Checking null values in each
print(df_test.isnull().sum())

for item in le_list:
    df_test[item] = le.fit_transform(df_test[item])
    
df_test.info()

test_list = np.arange(1,11)
X_test = df_test.iloc[:, test_list].values

X_test = X_test.astype(np.float32)


xgb = XGBRegressor(n_estimators=100, booster = 'gbtree' , learning_rate=0.035, gamma=50, subsample=0.5,
                           colsample_bytree= 1, max_depth=3, tree_method = 'hist')
xgb.fit(X,y)
y_pred_xgb = xgb.predict(X_test)
