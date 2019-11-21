#!/usr/bin/env python
# This script calculates effectiveness of all reference algorithms wrapped from
# skl and saves them to reference.csv.

import numpy as np
import os  # to list files
import re  # to use regex
import csv  # to save some outputratio
import json
from tqdm import tqdm
import ksienie as ks
from sklearn import neighbors, naive_bayes, svm, tree, neural_network
from sklearn import base
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing

# Initialize classifiers
classifiers = {
    "GNB": naive_bayes.GaussianNB(),
    "kNN": neighbors.KNeighborsClassifier(),
    'SVC': svm.SVC(gamma='scale'),
    'DTC': tree.DecisionTreeClassifier(),
    #'MLP': neural_network.MLPClassifier()
}

# Choose metrics
used_metrics = {
    "ACC": metrics.accuracy_score,
    "BAC": metrics.balanced_accuracy_score,
    #'APC': metrics.average_precision_score,
    #'BSL': metrics.brier_score_loss,
    #'CKS': metrics.cohen_kappa_score,
    #'F1': metrics.f1_score,
    #'HaL': metrics.hamming_loss,
    #'HiL': metrics.hinge_loss,
    #'JSS': metrics.jaccard_similarity_score,
    #'LoL': metrics.log_loss,
    #'MaC': metrics.matthews_corrcoef,
    #'PS': metrics.precision_score,
    #'RCS': metrics.recall_score,
    #'AUC': metrics.roc_auc_score,
    #'ZOL': metrics.zero_one_loss,
}

# Gather all the datafiles and filter them by tags
files = ks.dir2files("datasets/")
tag_filter = ["imbalanced"]  # , "multi-class"]
datasets = []
for file in files:
    X, y, dbname, tags = ks.csv2Xy(file)
    intersecting_tags = ks.intersection(tags, tag_filter)
    if len(intersecting_tags):
        datasets.append((X, y, dbname))

# Prepare results cube
print(
    "# Experiment on %i datasets, with %i estimators using %i metrics."
    % (len(datasets), len(classifiers), len(used_metrics))
)
rescube = np.zeros((len(datasets), len(classifiers), len(used_metrics), 5))

# Iterate datasets
for i, dataset in enumerate(tqdm(datasets, desc="DBS", ascii=True)):
    # load dataset
    X, y, dbname = dataset

    # Folds
    skf = model_selection.StratifiedKFold(n_splits=5)
    for fold, (train, test) in enumerate(
        tqdm(skf.split(X, y), desc="FLD", ascii=True, total=5)
    ):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        for c, clf_name in enumerate(tqdm(classifiers, desc="CLF", ascii=True)):
            clf = base.clone(classifiers[clf_name])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            for m, metric_name in enumerate(tqdm(used_metrics, desc="MET", ascii=True)):
                try:
                    score = used_metrics[metric_name](y_test, y_pred)
                    rescube[i, c, m, fold] = score
                except:
                    rescube[i, c, m, fold] = np.nan

np.save("results/rescube", rescube)
with open("results/legend.json", "w") as outfile:
    json.dump(
        {
            "datasets": [obj[2] for obj in datasets],
            "classifiers": list(classifiers.keys()),
            "metrics": list(used_metrics.keys()),
            "folds": 5,
        },
        outfile,
        indent="\t",
    )

print("\n")
