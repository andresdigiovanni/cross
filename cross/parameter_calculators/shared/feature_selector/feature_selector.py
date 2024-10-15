import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from cross.parameter_calculators.shared import evaluate_model
from cross.parameter_calculators.shared.permutation_feature_importance import (
    PermutationFeatureImportance,
)


class FeatureSelector:
    def fit(
        self, x, y, problem_type, maximize=True, transformer=None, early_stopping=None
    ):
        selected_features_idx = []
        best_score = float("-inf") if maximize else float("inf")
        features_added_without_improvement = 0

        base_score = evaluate_model(x, y, problem_type)

        # Permutation feature importance
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        if transformer:
            x_train = transformer.fit_transform(x_train, y_train)
            x_test = transformer.transform(x_test, y_test)

        model = self._get_model(problem_type)
        model.fit(x_train, y_train)
        permutation_feature_importance = PermutationFeatureImportance(
            model, metric=(accuracy_score if maximize else mean_squared_error)
        )
        permutation_feature_importance.fit(x_test, y_test)
        feature_importances = permutation_feature_importance.feature_importances_

        # Ordenar las características según la importancia
        indices = np.argsort(feature_importances)[::-1]

        # Seleccionar características una por una y comprobar la mejora
        for idx in indices:
            current_features_idx = selected_features_idx + [idx]
            score = evaluate_model(
                x, y, problem_type, transformer, columns_idx=current_features_idx
            )

            has_improved = (maximize and score > best_score) or (
                not maximize and score < best_score
            )
            if not has_improved:
                features_added_without_improvement += 1

                if (
                    early_stopping
                    and features_added_without_improvement >= early_stopping
                ):
                    break

                continue

            selected_features_idx.append(idx)
            best_score = score
            features_added_without_improvement = 0

            has_reach_base_score = (maximize and score >= base_score) or (
                not maximize and score <= base_score
            )
            if early_stopping is None and has_reach_base_score:
                break

        selected_features = [x_train.columns[i] for i in selected_features_idx]
        return best_score, selected_features

    def _get_model(self, problem_type):
        if problem_type == "classification":
            return RandomForestClassifier()

        elif problem_type == "regression":
            return RandomForestRegressor()

        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")
