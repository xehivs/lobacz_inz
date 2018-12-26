# A little module with some useful methods by Paweł Ksieniewicz.
import numpy as np
import os
import re

def csv2Xy(path):
    ds = np.genfromtxt(path, delimiter=',')
    X = ds[:,:-1]
    y = ds[:,-1].astype(int)
    return X, y

def dir2files(path, extention='csv'):
    return [path + x for \
        x in os.listdir(path) \
        if re.match('^([a-zA-Z0-9])+\.%s$' % extention, x)]
