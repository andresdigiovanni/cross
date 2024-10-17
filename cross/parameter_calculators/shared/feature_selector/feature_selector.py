import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score

from cross.transformations.utils.dtypes import numerical_columns


class FeatureSelector:
    def fit(self, x, y, model, scoring, direction, transformer=None, early_stopping=3):
        selected_features_idx = []
        best_score = float("-inf") if direction == "maximize" else float("inf")
        features_added_without_improvement = 0

        if transformer:
            x_transformed = transformer.fit_transform(x, y)

        else:
            x_transformed = x.copy()

        num_columns = numerical_columns(x_transformed)
        x_transformed = x_transformed.loc[:, num_columns]

        indices = self._feature_importance(model, x_transformed, y)

        # Seleccionar características una por una y comprobar la mejora
        for idx in indices:
            current_features_idx = selected_features_idx + [idx]

            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(
                model,
                x_transformed.iloc[:, current_features_idx],
                y,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
            )
            score = np.mean(scores)

            has_improved = (direction == "maximize" and score > best_score) or (
                direction != "maximize" and score < best_score
            )

            if has_improved:
                selected_features_idx.append(idx)
                best_score = score
                features_added_without_improvement = 0

            else:
                features_added_without_improvement += 1

                if features_added_without_improvement >= early_stopping:
                    break

        selected_features = [x_transformed.columns[i] for i in selected_features_idx]
        return selected_features

    def _feature_importance(self, model, x, y):
        model.fit(x, y)

        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_

        elif hasattr(model, "coef_"):
            feature_importances = model.coef_

        else:
            result = permutation_importance(model, x, y, n_repeats=5, random_state=42)
            feature_importances = result.importances_mean

        indices = np.argsort(feature_importances)[::-1]
        return indices
