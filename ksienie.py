# A little module with some useful methods by Paweł Ksieniewicz.
import numpy as np

def csv2Xy(path):
    ds = np.genfromtxt(path, delimiter=',')
    X = ds[:,:-1]
    y = ds[:,-1].astype(int)
    return X, y
