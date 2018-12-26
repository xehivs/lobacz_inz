#!/usr/bin/env python
# This script calculates effectiveness of all reference algorithms wrapped from
# skl and saves them to reference.csv.

import numpy as np
import os  # to list files
import re  # to use regex
import csv  # to save some output

# Gather all the datafiles
directory = 'datasets/'
files = [(directory + x, x[:-4]) for \
    x in os.listdir(directory) \
    if re.match('^([a-zA-Z0-9])+\.csv$', x)]

# Iterate datafiles
with open('reference.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    # write header
    writer.writerow(['dataset', 'knnacc', 'knnbac', 'gnbacc', 'gnbbac', 'dtcacc', 'dtcbac', 'mlpacc', 'mlpbac', 'svcacc', 'svcbac'])

    scores = np.zeros((11)).astype(int)
    for file in files:
        # Quick hack to ignore files with missing values
        if file[1] in exclusions:
            continue
        # load dataset
        dataset = Dataset(file[0])
        print dataset

        # initialize classifiers
        classifiers = [
            sklKNN(dataset, {'k': 5}),
            sklGNB(dataset, {}),
            sklDTC(dataset, {}),
            sklMLP(dataset, {}),
            sklSVC(dataset, {})
        ]

        # and save output
        row = [dataset.db_name]
        for classifier in classifiers:
            row.extend(classifier.quickLoop().values())
        bestAcc = (0,0)
        bestBac = (0,0)

        for i in xrange(5):
            acc = row[1 + 2*i]
            bac = row[2 + 2*i]

            if not np.isnan(acc):
                if bestAcc[1] < acc:
                    bestAcc = (i, acc)

            if not np.isnan(bac):
                if bestBac[1] < bac:
                    bestBac = (i, bac)

        for i in xrange(5):
            acc = row[1 + 2*i]
            bac = row[2 + 2*i]

            if np.isnan(acc):
                row[1 + 2*i] = '\\tiny NaN'
            else:
                if acc == bestAcc[1]:
                    scores[1+ 2*i] += 1
                    row[1 + 2*i] = '\\color{red} \\oldstylenums{%.3f}' % acc
                else:
                    row[1 + 2*i] = '\\oldstylenums{%.3f}' % acc

            if np.isnan(bac):
                row[2 + 2*i] = '\\tiny NaN'
            else:
                if bac == bestBac[1]:
                    scores[2+ 2*i] += 1
                    row[2 + 2*i] = '\\color{red} \\oldstylenums{%.3f}' % bac
                else:
                    row[2 + 2*i] = '\\oldstylenums{%.3f}' % bac

        writer.writerow(row)
    print scores
