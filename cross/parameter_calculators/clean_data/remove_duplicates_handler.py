from cross.transformations.clean_data import RemoveDuplicatesHandler


class RemoveDuplicatesParamCalculator:
    def calculate_best_params(self, x, y, problem_type, verbose):
        if self._has_duplicates(x):
            return self._get_remove_duplicates_params()

        return None

    def _has_duplicates(self, x):
        return x.duplicated().any()

    def _get_remove_duplicates_params(self):
        remove_duplicates_handler = RemoveDuplicatesHandler()
        return {
            "name": remove_duplicates_handler.__class__.__name__,
            "params": remove_duplicates_handler.get_params(),
        }
