# Import datasets from keel
import ksienie as ks
import numpy as np

datasets = ks.datasets_for_groups([
    "imb_IRhigherThan9p1",
    "imb_IRhigherThan9p2"
])

for i, dataset in enumerate(datasets):
    print(dataset[1])

    # Load dataset
    X, y, _, _ = ks.load_dataset(dataset)
    print(X.shape, y.shape)

    db = np.hstack((X, y[:, np.newaxis]))

    print(db.shape)

    filename = "datasets/%s.csv" % dataset[1]

    print(filename)

    np.savetxt(filename, db)
    #print(y.shape)
    #print(X_[0])
    #print(X_)
    #print(y_)

    #exit()
