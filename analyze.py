#!/usr/bin/env python
# A script to analyze datasets available at w4k2/benchmark_datasets.

import os  # to list files
import re  # to use regex
import csv  # to save some output
import numpy as np  # to calculate ratio
import ksienie as ks

# Gather all the datafiles
files = ks.dir2files('datasets/')

# Iterate datafiles
with open('datasets.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    # write header
    writer.writerow(['dataset', 'samples', 'features', 'classes', 'ratio', 'tags'])
    for file in files:
        # load dataset
        X, y = ks.csv2Xy(file)
        dbname = file.split('/')[-1].split('.')[0]

        # gather information
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

        ratio = str(max(ratio))
        tags = " ".join(tags)

        # write information
        writer.writerow([
            dbname,
            numberOfSamples,
            numberOfFeatures,
            numberOfClasses,
            ratio,
            tags])

        print("%3i features, %5i samples, %2i classes, %3s ratio - %s (%s)" % (
            numberOfFeatures,
            numberOfSamples,
            numberOfClasses,
            ratio,
            dbname,
            tags)
        )
