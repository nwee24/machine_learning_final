"""Custom transformer dùng trong pipeline (để load best_model.joblib cần import module này)."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ClusterAdder(BaseEstimator, TransformerMixin):
    """Thêm one-hot nhãn cụm (từ KMeans đã fit) vào ma trận đặc trưng."""
    def __init__(self, kmeans, behavior_idx, n_clusters):
        self.kmeans = kmeans
        self.behavior_idx = behavior_idx
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        c = self.kmeans.predict(X[:, self.behavior_idx])
        return np.hstack([X, np.eye(self.n_clusters)[c]])
