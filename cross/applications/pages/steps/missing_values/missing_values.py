class MissingValuesBase:
    def __init__(self):
        self.ACTIONS_ALL = {
            "Do nothing": "none",
            "Fill with 0": "fill_0",
            "Fill with mode": "fill_mode",
        }
        self.ACTIONS_CAT = {
            "Most frequent": "most_frequent",
        }
        self.ACTIONS_NUM = {
            "Fill with mean": "fill_mean",
            "Fill with median": "fill_median",
            "KNN imputation": "fill_knn",
        }
        self.ACTIONS = self.ACTIONS_ALL | self.ACTIONS_CAT | self.ACTIONS_NUM
