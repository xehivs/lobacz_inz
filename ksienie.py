# A little module with some useful methods by Paweł Ksieniewicz.
import numpy as np
import os
import re
import json

def json2object(path):
    with open(path) as json_data:
        return json.load(json_data)

def csv2Xy(path):
    ds = np.genfromtxt(path, delimiter=',')
    X = ds[:,:-1]
    y = ds[:,-1].astype(int)
    dbname = path.split('/')[-1].split('.')[0]
    tags = tags4Xy(X, y)
    return X, y, dbname, tags

def dir2files(path, extention='csv'):
    return [path + x for \
        x in os.listdir(path) \
        if re.match('^([a-zA-Z0-9])+\.%s$' % extention, x)]

def tags4Xy(X, y):
    tags = []
    numberOfFeatures = X.shape[1]
    numberOfSamples = len(y)
    numberOfClasses = len(np.unique(y))
    if numberOfClasses == 2:
        tags.append("binary")
    else:
        tags.append("multi-class")
    if numberOfFeatures >= 8:
        tags.append("multi-feature")

    # Calculate ratio
    ratio = [0.] * numberOfClasses
    for y_ in y:
        ratio[y_] += 1
    ratio = [int(round(i / min(ratio))) for i in ratio]
    if max(ratio) > 4:
        tags.append("imbalanced")

    return tags

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
