import numpy as np
from sklearn.metrics import accuracy_score


class PermutationFeatureImportance:
    def __init__(self, model, metric=accuracy_score, n_repeats=5, random_state=None):
        self.model = model
        self.metric = metric
        self.n_repeats = n_repeats
        self.feature_importances_ = None
        self.rng = np.random.default_rng(random_state)

    def fit(self, x, y):
        baseline_score = self.metric(y, self.model.predict(x))
        self.feature_importances_ = np.zeros(x.shape[1])

        for col in range(x.shape[1]):
            shuffle_scores = np.zeros(self.n_repeats)

            for n in range(self.n_repeats):
                x_shuffle = x.copy()
                x_shuffle.iloc[:, col] = self.rng.permutation(x_shuffle.iloc[:, col])

                shuffle_score = self.metric(y, self.model.predict(x_shuffle))
                shuffle_scores[n] = baseline_score - shuffle_score

            self.feature_importances_[col] = np.mean(shuffle_scores)
