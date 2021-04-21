import numpy as np
from config import *
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

def load_data():
    X_train = np.loadtxt("./X_train.csv", delimiter=",")
    y_train = np.loadtxt("./y_train.csv", delimiter=",")
    X_val = np.loadtxt("./X_val.csv", delimiter=",")
    y_val = np.loadtxt("./y_val.csv", delimiter=",")
    return X_train, y_train, X_val, y_val

def scale_data(df1, df2):
    scaler = MinMaxScaler().fit(np.append(df1, df2, axis=0))
    df1 = scaler.transform(df1)
    df2 = scaler.transform(df2)
    return df1, df2