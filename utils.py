import numpy as np
from config import *
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

chi2=[0,2,3,8,10,11,17,50,58,64,66,67,68,72,74,75,76,81,82,83,90,91,114,115,122,124]
gini=[3,8,9,11,15,16,19,24,30,33,35,37,57,66,68,69,74,75,77,78,80,81,83,88,91,92,96,97,100,104,112,113,115,123,124]


def load_data():
    X_train = np.loadtxt("./X_train.csv", delimiter=",")
    y_train = np.loadtxt("./y_train.csv", delimiter=",")
    X_val = np.loadtxt("./X_val.csv", delimiter=",")
    y_val = np.loadtxt("./y_val.csv", delimiter=",")
    X_train, X_val = scale_data(X_train,X_val)
    return X_train, y_train, X_val, y_val

def scale_data(df1, df2):
    scaler = MinMaxScaler().fit(np.append(df1, df2, axis=0))
    df1 = scaler.transform(df1)
    df2 = scaler.transform(df2)
    return df1, df2

def feature_selected(df,columns):
    return df[:,columns]