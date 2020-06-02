# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:33:28 2020

@author: ENES YILDIZ 1.DÖNEM 0101190186

"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import plotting
from scipy import stats
from pandas import DataFrame 
plt.style.use("ggplot")
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
from sklearn import tree
import sklearn


kel = pd.read_csv("diabetes.csv")
print(kel)

f,ax=plt.subplots(figsize = (18,18))
sns.heatmap(kel.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.savefig('graph.png')
plt.show()

"""
Veri setinde korrealasyon oluşturacak değer kümeleri yok olduğunu grafikten gördük
"""

x = kel.iloc[: ,:-1].values
y = kel.iloc[:,-1]


fonksiyon = StandardScaler()

x_1 = fonksiyon.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3)

fonksiyon2 = tree.DecisionTreeClassifier()

fonksiyon3 = fonksiyon2.fit(x_train,y_train)

Tahmin = fonksiyon3.predict(x_test)

print(classification_report(y_test, Tahmin ))
