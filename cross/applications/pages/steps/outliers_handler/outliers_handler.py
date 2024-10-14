class OutliersHandlingBase:
    actions = {
        "Do nothing": "none",
        "Cap to threshold": "cap",
        "Replace with median": "median",
    }
    detection_methods = {
        "IQR": "iqr",
        "Z-score": "zscore",
        "Local Outlier Factor": "lof",
        "Isolation Forest": "iforest",
    }
