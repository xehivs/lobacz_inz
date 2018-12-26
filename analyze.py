#!/usr/bin/env python
# A script to analyze datasets available at ksskml/data.

import os  # to list files
import re  # to use regex
import csv  # to save some output
import numpy as np  # to calculate ratio
import weles as wl  # to analyze with ksskml

# Gather all the datafiles
directory = 'datasets/'
files = [(directory + x, x[:-4]) for \
    x in os.listdir(directory) \
    if re.match('^([a-zA-Z0-9])+\.csv$', x)]

# Iterate datafiles
with open('datasets.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    # write header
    writer.writerow(['dataset', 'samples', 'features', 'classes', 'ratio', 'tags'])
    for file in files:
        # load dataset
        dataset = wl.Dataset(file[0])

        # gather information
        tags = []
        numberOfFeatures = dataset.features
        numberOfSamples = len(dataset.samples)
        numberOfClasses = len(dataset.classes)
        if numberOfClasses == 2:
            tags.append("binary")
        else:
            tags.append("multi-class")
        if numberOfFeatures >= 8:
            tags.append("multi-feature")
        # Calculate ratio
        ratio = [0.] * numberOfClasses
        for sample in dataset.samples:
            ratio[sample.label] += 1
        ratio = [int(round(i / min(ratio))) for i in ratio]
        if max(ratio) > 4:
            tags.append("imbalanced")

        ratio = str(max(ratio))
        tags = " ".join(tags)

        # write information
        writer.writerow([
            dataset.db_name,
            numberOfSamples,
            numberOfFeatures,
            numberOfClasses,
            ratio,
            tags])

        print "%2i features, %5i samples, %2i classes, %s ratio - %s (%s)" % (
            numberOfFeatures,
            numberOfSamples,
            numberOfClasses,
            ratio,
            dataset,
            tags)
