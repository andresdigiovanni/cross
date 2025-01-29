from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score

from cross.auto_parameters.shared.feature_selector.shared import feature_importance


class RecursiveFeatureAddition:
    def __init__(
        self,
        model,
        scoring: str,
        direction: str = "maximize",
        cv: Union[int, Callable] = 5,
        groups: Optional = None,
        early_stopping: int = 3,
        target_score: bool = True,
        verbose: bool = False,
    ):
        """
        Initializes the RecursiveFeatureAddition class.

        Args:
            model: Machine learning model with a fit method.
            scoring (str): Scoring metric for evaluation.
            direction (str, optional): "maximize" to increase score or "minimize" to decrease. Defaults to "maximize".
            cv (Union[int, Callable]): Number of cross-validation folds or a custom cross-validation generator.
            groups (Optional): Group labels for cross-validation splitting.
            early_stopping (int, optional): Maximum number of non-improving additions. Defaults to 3.
            target_score (bool, optional): Whether to use the full-feature score as a stopping criterion. Defaults to True.
            verbose (bool, optional): Whether to print progress messages. Defaults to False.
        """
        self.model = model
        self.scoring = scoring
        self.direction = direction
        self.cv = cv
        self.groups = groups
        self.early_stopping = early_stopping
        self.target_score = target_score
        self.verbose = verbose

        self.scores_history = []
        self.target_score_value = None
        self.selected_features_names = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> list:
        """
        Recursively adds features based on their importance and evaluates performance.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target variable.

        Returns:
            list: List of selected feature names.
        """
        X = X.copy()

        self.model.fit(X, y)
        scores = cross_val_score(
            self.model,
            X,
            y,
            scoring=self.scoring,
            cv=self.cv,
            groups=self.groups,
            n_jobs=-1,
        )
        self.target_score_value = np.mean(scores)
        if self.verbose:
            print(f"Target score: {self.target_score_value}")

        feature_importances = feature_importance(self.model, X, y)
        feature_indices = np.argsort(feature_importances)[::-1]

        selected_features_idx = self._evaluate_features(X, y, feature_indices)
        self.selected_features_names = [X.columns[i] for i in selected_features_idx]

        return self.selected_features_names

    def _evaluate_features(
        self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray
    ) -> list:
        best_score = float("-inf") if self.direction == "maximize" else float("inf")
        selected_features_idx = []
        features_added_without_improvement = 0

        for i, idx in enumerate(feature_indices):
            current_features_idx = selected_features_idx + [idx]

            scores = cross_val_score(
                self.model,
                X.iloc[:, current_features_idx],
                y,
                scoring=self.scoring,
                cv=self.cv,
                groups=self.groups,
                n_jobs=-1,
            )
            score = np.mean(scores)

            if self._is_score_improved(score, best_score, self.direction):
                self.scores_history.append(score)
                selected_features_idx.append(idx)
                best_score = score
                features_added_without_improvement = 0

                if self.verbose:
                    print(
                        f"{i + 1}/{len(feature_indices)} Added {X.columns[idx]} with score {score}"
                    )
            else:
                features_added_without_improvement += 1
                if self.verbose:
                    print(f"{i + 1}/{len(feature_indices)} Skipped {X.columns[idx]}")

            if self.target_score and not self._is_score_improved(
                score, self.target_score_value, self.direction
            ):
                continue

            if features_added_without_improvement >= self.early_stopping:
                break

        return selected_features_idx

    def _is_score_improved(
        self, score: float, best_score: float, direction: str
    ) -> bool:
        return (direction == "maximize" and score > best_score) or (
            direction == "minimize" and score < best_score
        )

    def create_figure_progression(self) -> plt.Figure:
        """
        Plots the score progression as features are added.

        Returns:
            plt.Figure: The generated figure.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            range(1, len(self.scores_history) + 1),
            self.scores_history,
            marker="o",
            label="Feature Addition Progress",
        )
        ax.axhline(
            y=self.target_score_value,
            color="r",
            linestyle="--",
            label="Target Score",
        )
        ax.set_xticks(range(1, len(self.selected_features_names) + 1))
        ax.set_xticklabels(self.selected_features_names, rotation=45, ha="right")
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("Score")
        ax.set_title("Score Progression by Feature Addition")
        ax.legend()
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier

    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    model = KNeighborsClassifier()
    rfa = RecursiveFeatureAddition(model, scoring="roc_auc_ovr", verbose=True)
    selected_features = rfa.fit(X, y)
    print("Selected Features:", selected_features)

    fig = rfa.create_figure_progression()
    img_path = "rfa_features_scores.png"
    fig.savefig(img_path, format="png")
