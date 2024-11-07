from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from cross.transformations.feature_engineering import CorrelatedSubstringEncoder
from cross.transformations.utils.dtypes import categorical_columns


class CorrelatedSubstringEncoderParamCalculator:
    MIN_FREQ = 0.05
    N = 6
    MIN_UNIQUE_VALUES = 10
    CORRELATION_THRESHOLD = 0.01
    BEAM_WIDTH = 15

    def calculate_best_params(self, x, y, model, scoring, direction, verbose):
        columns = categorical_columns(x)
        substring_dicts = {}

        for column in tqdm(columns, disable=not verbose):
            num_unique_values = x[column].nunique()

            # Skip columns with unique values below the minimum threshold
            if num_unique_values < self.MIN_UNIQUE_VALUES:
                continue

            # Extract and evaluate substrings
            substrings = self._extract_substrings(x, column)
            substring_scores = self._calculate_substrings_scores(
                x, y, substrings, column, direction
            )

            # Perform substring combination search
            selected_substrings = self._beam_search_best_substrings(
                x, y, column, substring_scores, direction
            )

            # Save selected substrings if valid combinations are found
            if selected_substrings:
                substring_dicts[column] = selected_substrings

        if substring_dicts:
            substring_encoder = CorrelatedSubstringEncoder(substrings=substring_dicts)
            return {
                "name": substring_encoder.__class__.__name__,
                "params": substring_encoder.get_params(),
            }
        return None

    def _extract_substrings(self, x, column):
        substrings = set()
        for row in x[column]:
            words = str(row).split()
            for word in words:
                for length in range(1, self.N + 1):
                    substrings.update(
                        word[i : i + length] for i in range(len(word) - length + 1)
                    )
        return list(substrings)

    def _calculate_substrings_scores(self, x, y, substrings, column, direction):
        substring_scores = []
        max_occurrences = len(x)
        min_occurrences = int(len(x) * self.MIN_FREQ)
        new_column = f"{column}__temp"

        for substring in substrings:
            # Check if the substring is present in each entry
            x[new_column] = x[column].apply(lambda val: 1 if substring in val else 0)
            occurrences = (x[new_column] == 1).sum()

            # Filter substrings with occurrences within the specified range
            if max_occurrences > occurrences >= min_occurrences:
                corr = self._evaluate_correlation(x, y, new_column, direction)
                if corr >= self.CORRELATION_THRESHOLD:
                    substring_scores.append((substring, corr))

            x.drop(columns=[new_column], inplace=True)

        # Sort substrings by correlation score in descending order
        substring_scores.sort(key=lambda x: x[1], reverse=True)
        return substring_scores

    def _beam_search_best_substrings(self, x, y, column, substring_scores, direction):
        new_column = f"{column}__corr_substring"

        # Initialize pending combinations with empty combinations to expand in the first iteration
        pending_combinations = [[]]
        tested_combinations = set()
        all_combinations = []

        continue_testing = True

        while continue_testing:
            continue_testing = False

            # Expand each pending combination
            for combination in pending_combinations:
                n_substrings_tested = 0

                for substring, _ in substring_scores:
                    # Skip substrings already included in the current combination
                    if substring in combination:
                        continue

                    # Create a new combination by adding the current substring
                    updated_combination = combination + [substring]

                    # Skip if the combination was already tested
                    if tuple(updated_combination) in tested_combinations:
                        continue

                    # Mark combination as tested
                    tested_combinations.add(tuple(updated_combination))

                    continue_testing = True

                    # Apply the transformation with the current substring combination
                    encoder = CorrelatedSubstringEncoder(
                        substrings={column: updated_combination}
                    )
                    transformed_x = encoder.transform(x[[column]])

                    # Encode unique values for the transformed data
                    label_encoder = LabelEncoder()
                    transformed_x[new_column] = label_encoder.fit_transform(
                        transformed_x[new_column]
                    )

                    # Calculate the correlation for the current combination
                    new_corr = self._evaluate_correlation(
                        transformed_x, y, new_column, direction
                    )

                    all_combinations.append((updated_combination, new_corr))

                    n_substrings_tested += 1
                    if n_substrings_tested >= self.BEAM_WIDTH:
                        break

            # Sort combinations by score and select the top BEAM_WIDTH
            all_combinations.sort(key=lambda x: x[1], reverse=True)
            pending_combinations = [
                combo[0] for combo in all_combinations[: self.BEAM_WIDTH]
            ]

        # Return the best combination found
        return all_combinations[0][0]

    def _evaluate_correlation(self, x, y, column, direction):
        corr_func = (
            mutual_info_classif if direction == "maximize" else mutual_info_regression
        )
        corr = corr_func(x[[column]], y).item()
        return corr
