
Code written by: Kang Yoo Seong and Russlan Jaafreh

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from google.colab import files

myfile = files.upload()
df1 = pd.read_excel("Full feature file.csv") #please write the full feature file 
Before_Features = df1.drop(['Compound','Hardness',], axis =1)

#Variance threshold 0.1 analysis
from sklearn.feature_selection import VarianceThreshold
var_thres = VarianceThreshold(threshold=0.01)
var_thres.fit(Before_Features)
var_thres.get_support()
constant_columns = [column for column in Before_Features.columns if column not in Before_Features.columns[var_thres.get_support()]]
After_Variance = Before_Features.drop(constant_columns,axis=1)

#Pearson Correlation 

#this function was extracted from Krish Naik's github: https://github.com/krishnaik06/Complete-Feature-Selection/blob/master/2-Feature%20Selection-%20Correlation.ipynb
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    af_corr = dataset.drop(col_corr,axis=1)
    return af_corr

#Pearson Correlation > 0.7
af_both = correlation(After_Variance, 0.7)
af_both.shape
A_cor = af_both.corr()
plt.figure(figsize=(25,20))

#Heatmap
sns.heatmap(A_cor,cmap=plt.cm.CMRmap_r,annot=False)
#plt.title("Before_Correlation",size = 20)
plt.show()

af_data = pd.concat([af_both,df1[["Compound","Hardness"]]], axis=1)

af_data.to_csv("Data fter feature Preprocessing.csv")