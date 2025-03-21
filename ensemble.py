import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors

class VotingAnomalyDetector:
    """
    Classe qui entraîne :
      - IsolationForest
      - One-Class SVM
      - "DBSCAN-like" via NearestNeighbors
    Et applique un vote majoritaire (≥2 sur 3) pour détecter les anomalies.

    Méthodes:
      - fit(X)
      - predict(X) -> 0=normal, 1=anomalie
    """

    def __init__(self, if_contamination=0.01, svm_nu=0.01,
                 threshold_knn=0.6, n_neighbors=10, random_state=42):
        self.if_contamination = if_contamination
        self.svm_nu = svm_nu
        self.threshold_knn = threshold_knn
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        self.isolation_forest = None
        self.svm = None
        self.neigh = None

    def fit(self, X):
        # 1) Isolation Forest
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=self.if_contamination,
            random_state=self.random_state
        )
        self.isolation_forest.fit(X)

        # 2) One-Class SVM
        self.svm = OneClassSVM(
            kernel="rbf",
            nu=self.svm_nu,
            gamma="scale"
        )
        self.svm.fit(X)

        # 3) DBSCAN-like => NearestNeighbors
        self.neigh = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.neigh.fit(X)

    def predict(self, X):
        # IsolationForest => -1 = anomalie => 1
        y_if = self.isolation_forest.predict(X)
        y_if = np.where(y_if == -1, 1, 0)

        # One-Class SVM => -1 = anomalie => 1
        y_svm = self.svm.predict(X)
        y_svm = np.where(y_svm == -1, 1, 0)

        # DBSCAN-like => distance moyenne aux k plus proches voisins
        distances = self.neigh.kneighbors(X)[0]
        z_predict = distances.mean(axis=1)
        y_knn = np.where(z_predict > self.threshold_knn, 1, 0)

        # Vote majoritaire => ≥2 => anomalie
        votes = y_if + y_svm + y_knn
        y_vote = np.where(votes >= 2, 1, 0)

        return y_vote
