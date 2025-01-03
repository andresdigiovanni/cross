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
        cv: int = 5,
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
            cv (int, optional): Number of cross-validation folds. Defaults to 5.
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
            model,
            X,
            y,
            feature_indices,
            scoring,
            cv,
            direction,
            early_stopping,
        )

        return [X.columns[i] for i in selected_features_idx]

    @staticmethod
    def _evaluate_features(
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
        scoring: str,
        cv: int,
        direction: str,
        early_stopping: int,
    ) -> list:
        """
        Evaluates features and returns the indices of selected features.

        Args:
            model: Machine learning model with a fit method.
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target variable.
            feature_indices (np.ndarray): Indices of features sorted by importance.
            scoring (str): Scoring metric for evaluation.
            cv (int): Number of cross-validation folds.
            direction (str): "maximize" to increase score or "minimize" to decrease.
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
