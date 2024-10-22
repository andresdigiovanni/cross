class OutliersHandlingBase:
    ACTIONS = {
        "Do nothing": "none",
        "Cap to threshold": "cap",
        "Replace with median": "median",
    }
    DETECTION_METHODS = {
        "IQR": "iqr",
        "Z-score": "zscore",
        "Local Outlier Factor": "lof",
        "Isolation Forest": "iforest",
    }
