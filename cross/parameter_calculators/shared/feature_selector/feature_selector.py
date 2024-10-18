from typing import List

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score

from cross.transformations.utils.dtypes import numerical_columns


class FeatureSelector:
    def fit(
        self,
        x,
        y,
        model,
        scoring,
        direction="maximize",
        transformer=None,
        cv=5,
        early_stopping=5,
    ):
        x_transformed = self._apply_transformer(x, y, transformer)
        num_columns = numerical_columns(x_transformed)
        x_transformed = x_transformed.loc[:, num_columns]

        feature_indices = self._feature_importance(model, x_transformed, y)

        # Evaluate features and select those that improve performance
        selected_features_idx = self._evaluate_features(
            model,
            x_transformed,
            y,
            feature_indices,
            scoring,
            cv,
            direction,
            early_stopping,
        )

        return [x_transformed.columns[i] for i in selected_features_idx]

    def _apply_transformer(self, x, y, transformer):
        if transformer:
            return transformer.fit_transform(x, y)

        return x.copy()

    def _feature_importance(self, model, x, y):
        model.fit(x, y)

        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_

        elif hasattr(model, "coef_"):
            feature_importances = model.coef_

        else:
            result = permutation_importance(model, x, y, n_repeats=5, random_state=42)
            feature_importances = result.importances_mean

        return np.argsort(feature_importances)[::-1]

    def _evaluate_features(
        self,
        model,
        x_transformed,
        y,
        feature_indices,
        scoring,
        cv,
        direction,
        early_stopping,
    ):
        best_score = float("-inf") if direction == "maximize" else float("inf")

        selected_features_idx: List[int] = []
        features_added_without_improvement = 0

        for idx in feature_indices:
            current_features_idx = selected_features_idx + [idx]

            cv_split = KFold(n_splits=cv, shuffle=True, random_state=42)
            scores = cross_val_score(
                model,
                x_transformed.iloc[:, current_features_idx],
                y,
                scoring=scoring,
                cv=cv_split,
                n_jobs=-1,
            )
            score = np.mean(scores)

            has_improved = (direction == "maximize" and score > best_score) or (
                direction == "minimize" and score < best_score
            )

            if has_improved:
                selected_features_idx.append(idx)
                best_score = score
                features_added_without_improvement = 0
            else:
                features_added_without_improvement += 1

                if features_added_without_improvement >= early_stopping:
                    break

        return selected_features_idx
