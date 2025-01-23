from typing import Callable, Optional, Union

import numpy as np
from sklearn.model_selection import KFold, cross_val_score

from .shared import feature_importance


class RecursiveFeatureAddition:
    @staticmethod
    def fit(
        X: np.ndarray,
        y: np.ndarray,
        model,
        scoring: str,
        direction: str = "maximize",
        cv: Union[int, Callable] = 5,
        groups: Optional = None,
        early_stopping: int = 3,
    ) -> list:
        """
        Recursively adds features based on their importance and evaluates performance.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target variable.
            model: Machine learning model with a fit method.
            scoring (str): Scoring metric for evaluation.
            direction (str, optional): "maximize" to increase score or "minimize" to decrease. Defaults to "maximize".
            cv (Union[int, Callable]): Number of cross-validation folds or a custom cross-validation generator.
            groups (Optional): Group labels for cross-validation splitting.
            early_stopping (int, optional): Maximum number of non-improving additions. Defaults to 3.

        Returns:
            list: List of selected feature names.
        """
        X = X.copy()

        model.fit(X, y)
        feature_importances = feature_importance(model, X, y)
        feature_indices = np.argsort(feature_importances)[::-1]

        # Evaluate features and select those that improve performance
        selected_features_idx = RecursiveFeatureAddition._evaluate_features(
            X,
            y,
            model,
            feature_indices,
            scoring,
            direction,
            cv,
            groups,
            early_stopping,
        )

        return [X.columns[i] for i in selected_features_idx]

    @staticmethod
    def _evaluate_features(
        X: np.ndarray,
        y: np.ndarray,
        model,
        feature_indices: np.ndarray,
        scoring: str,
        direction: str,
        cv: Union[int, Callable],
        groups: Optional,
        early_stopping: int,
    ) -> list:
        """
        Evaluates features and returns the indices of selected features.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target variable.
            model: Machine learning model with a fit method.
            feature_indices (np.ndarray): Indices of features sorted by importance.
            scoring (str): Scoring metric for evaluation.
            direction (str): "maximize" to increase score or "minimize" to decrease.
            cv (Union[int, Callable]): Number of cross-validation folds or a custom cross-validation generator.
            groups (Optional): Group labels for cross-validation splitting.
            early_stopping (int): Maximum number of non-improving additions.

        Returns:
            list: Indices of selected features.
        """
        best_score = float("-inf") if direction == "maximize" else float("inf")

        selected_features_idx = []
        features_added_without_improvement = 0

        for idx in feature_indices:
            current_features_idx = selected_features_idx + [idx]

            cv_split = KFold(n_splits=cv, shuffle=True, random_state=42)
            scores = cross_val_score(
                model,
                X.iloc[:, current_features_idx],
                y,
                scoring=scoring,
                cv=cv_split,
                groups=groups,
                n_jobs=-1,
            )
            score = np.mean(scores)

            if RecursiveFeatureAddition._is_score_improved(
                score, best_score, direction
            ):
                selected_features_idx.append(idx)
                best_score = score
                features_added_without_improvement = 0

            else:
                features_added_without_improvement += 1

                if features_added_without_improvement >= early_stopping:
                    break

        return selected_features_idx

    @staticmethod
    def _is_score_improved(score: float, best_score: float, direction: str) -> bool:
        """
        Checks if the new score improves over the best score.

        Args:
            score (float): Current score.
            best_score (float): Best score so far.
            direction (str): "maximize" or "minimize".

        Returns:
            bool: True if the score is improved, False otherwise.
        """
        return (direction == "maximize" and score > best_score) or (
            direction == "minimize" and score < best_score
        )
