class CategoricalEncodingBase:
    ENCODING_OPTIONS = {
        "Do nothing": "none",
        "Label Encoder": "label",
        "Ordinal Encoder": "ordinal",
        "One Hot Encoder": "onehot",
        "Dummy Encoder": "dummy",
        "Binary Encoder": "binary",
        "Count Encoder": "count",
        "Target Encoder": "target",
        "Leave One Out Encoder": "loo",
        "Rank Hot Encoder": "rankhot",
    }
