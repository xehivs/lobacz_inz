# Undersampled Ensemble Support Vector Classifier
from sklearn.svm import SVC
from sklearn.base import clone, ClassifierMixin, BaseEstimator
from imblearn.over_sampling import SMOTE
import numpy as np


class ESSVC(ClassifierMixin, BaseEstimator):
    def __init__(self, n=10, random_state=42):
        self.n = n  # Liczba klasyfikatorów w puli
        self.random_state = random_state

    def fit(self, X, y):
        self.X_, self.y_ = np.copy(X), np.copy(y)
        self.ensemble_ = []
        for i in range(self.n):
            #print(np.unique(self.y_, return_counts=True))

            k = 5
            a = np.min(np.unique(self.y_, return_counts=True)[1])
            if a <= 5:
                #print("STOP!")
                k = a-1

            smote = SMOTE(random_state=i + self.random_state, k_neighbors=k)
            X_res, y_res = smote.fit_resample(self.X_, self.y_)

            clf = SVC(gamma="scale")
            clf.fit(X_res, y_res)

            self.ensemble_.append(clf)

    def predict(self, X):
        # Macierz wsparć zespołu
        return np.sum(np.array([clf.decision_function(X) for clf in self.ensemble_]), axis=0) > 0
