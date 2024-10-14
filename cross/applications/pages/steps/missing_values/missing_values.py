class MissingValuesBase:
    def __init__(self):
        self.actions_all = {
            "Do nothing": "none",
            "Fill with 0": "fill_0",
            "Fill with mode": "fill_mode",
        }
        self.actions_cat = {
            "Most frequent": "most_frequent",
        }
        self.actions_num = {
            "Fill with mean": "fill_mean",
            "Fill with median": "fill_median",
            "Interpolate": "interpolate",
            "KNN imputation": "fill_knn",
        }
        self.actions = self.actions_all | self.actions_cat | self.actions_num
