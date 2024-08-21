#Code Written by: Russlan Jaafreh

#Importing Pandas & Numpy
import pandas as pd
import numpy as np

#Importing data after VT and PC feature filtering
df = pd.read_csv('VP01.csv')
df.head()
df.describe()

#Importing full data to obtain the hardness values 
df2 = pd.read_csv('Full Data.csv')
Y=df2['Hardness']

#Scaling the data using a standard scaler
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
df_arr = scaler.fit_transform(df)
df1 = pd.DataFrame(df_arr)

#Splitting the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2, random_state=42)

#Imporitng the XGBoost regressor (Unoptimized)
from xgboost import XGBRegressor
model_xgb_un = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
model_xgb_un.fit(X_train,y_train)
R1 = model_xgb_un.score(X_test,y_test)

y_predict_xgb_un = model_xgb_un.predict(X_test)

#Imporitng the XGBoost regressor (Optimized)
from xgboost import XGBRegressor
model_xgb_op = XGBRegressor(base_score=0.5,booster='gbtree',gamma = 0, n_estimators=500, max_depth=4, eta=0.05, subsample=0.7, colsample_bytree=0.8, reg_alpha = 0.005)

model_xgb_op.fit(X_train,y_train)
R2= model_xgb_op.score(X_test,y_test)
y_predict_xgb_op = model_xgb_op.predict(X_test)


#Prediction
d_predict = pd.read_csv('Prediction_file.csv') ##please insert the file including the 272 feature (including load and crystal structure features obtained from magpie)
d_predict = d_predict[df.columns] # getting the 36 important feature we have obtained by VT and PC
d_predict_scaled = scaler.transform(d_predict)
df_scaled_predict = pd.DataFrame(d_predict_scaled)
y_predict = model_xgb_op.predict(df_scaled_predict)
y_predict_excel = pd.DataFrame(y_predict)
y_predict_excel.to_csv("Hardness_prediction.csv") ##make sure to specificy a directory
