#!/usr/bin/env python
import ksienie as ks
import numpy as np
from scipy import stats
import latextabs as lt

# Parameters
used_test = stats.ranksums
used_p = .05

# Load results
legend = ks.json2object('results/legend.json')
datasets = legend['datasets']
classifiers = legend['classifiers']
metrics = legend['metrics']
folds = legend['folds']
rescube = np.load('results/rescube.npy')

# First generate tables for each metric
for mid, metric in enumerate(metrics):
    print("Metric %s [%i]" % (metric, mid))

    table_file = open("results/tab_%s.tex" % metric, "w")
    table_file.write(lt.header4classifiers(classifiers))

    for did, dataset in enumerate(datasets):
        print("| Dataset %10s [%i]" % (dataset, did))

        # Subtable is 2d (clf, fold)
        subtable = rescube[did, :, mid, :]

        # Check if metric was valid
        if np.isnan(subtable).any():
            print("Unvaild")
            continue

        # Scores as mean over folds
        scores = np.mean(subtable, axis=1)
        stds = np.std(subtable, axis=1)

        # Get leader and check dependency
        dependency = np.zeros(len(classifiers)).astype(int)
        leader = np.argmax(scores)

        a = subtable[leader]
        for cid, clf in enumerate(classifiers):
            b = subtable[cid]
            test = used_test(a, b)
            # If pvalue is smaller or equal to used p, then
            # samples ar LIKELY drawn from different distributions
            # If pvalue is larger than used p, samples are LIKELY
            # Drawn from the same distribution
            dependency[cid] = test.pvalue > used_p

        print(dependency)
        print(scores)
        print(stds)
        table_file.write(lt.row(dataset,
                                dependency,
                                scores, stds))
    table_file.write(lt.footer("Results for %s metric" % metric))
    table_file.close()
