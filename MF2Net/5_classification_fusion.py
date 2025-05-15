import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import mode

class FusionClassifier:
    def __init__(self):
        self.svm = make_pipeline(StandardScaler(), SVC(probability=True))
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X, y):
        self.svm.fit(X, y)
        self.rf.fit(X, y)

    def predict(self, X):
        pred_svm = self.svm.predict(X)
        pred_rf = self.rf.predict(X)
        # Majority voting
        predictions = np.vstack((pred_svm, pred_rf)).T
        fused = mode(predictions, axis=1)[0].flatten()
        return fused

    def predict_proba_avg(self, X):
        proba_svm = self.svm.predict_proba(X)
        proba_rf = self.rf.predict_proba(X)
        return (proba_svm + proba_rf) / 2
