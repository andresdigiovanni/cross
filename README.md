![UI Cross](assets/logo.png)

-----------------

# Cross: a versatile toolkit for data preprocessing and feature engineering in machine learning

![PyPI version](https://img.shields.io/pypi/v/cross_ml)
![Downloads](https://img.shields.io/pypi/dm/cross_ml)

Cross is a Python library for data processing to train machine learning models, featuring scaling, normalization, feature creation through binning, and various mathematical operations between columns. It includes a graphical interface for exploring and generating transformations, with the option to export and reuse them.

![UI Cross](assets/ui_outliers_handling.png)

- [Getting Started](#getting-started)
- [Example of Use](#example-of-use)
  - [Define transformations](#define-transformations)
  - [Import Transformations from UI](#import-transformations-from-ui)
  - [Save and Load Transformations](#save-and-load-transformations)
  - [Auto transformations](#auto-transformations)
- [Transformations](#transformations)

  - [Clean Data](#clean-data)
    - [Missing Values](#missing-values)
  - [Numerical Features](#numerical-features)
    - [Handle Outliers](#handle-outliers)
    - [Non-Linear Transformation](#non-linear-transformation)
    - [Quantile Transformations](#quantile-transformations)
    - [Scale Transformations](#scale-transformations)
    - [Normalization](#normalization)
    - [Numerical Binning](#numerical-binning)
    - [Mathematical Operations](#mathematical-operations)
  - [Categorical Features](#categorical-features)
    - [Categorical Encoding](#categorical-encoding)
  - [Datetime Features](#datetime-features)
    - [Date Time Transforms](#date-time-transforms)
  - [Periodic Features](#periodic-features)
    - [Cyclical Features Transforms](#cyclical-features-transforms)
  - [Features Reduction](#features-reduction)
    - [Column Selection](#column-selection)
    - [Dimensionality Reduction](#dimensionality-reduction)


## Getting Started

To install the Cross library, run the following command:

```bash
pip install cross_ml
```

You can initialize the graphical interface with the command:

```bash
cross run
```

Access the application by navigating to `http://localhost:8501` in your browser.


## Example of Use

### Manual transformations

```python
from cross import CrossTransformer
from cross.transformations import (
    MathematicalOperations,
    NumericalBinning,
    OutliersHandler,
    ScaleTransformation,
)

# Define transformations
transformations = [
    OutliersHandler(
        handling_options={
            "sepal length (cm)": ("median", "iqr"),
            "sepal width (cm)": ("cap", "zscore"),
        },
        thresholds={
            "sepal length (cm)": 1.5,
            "sepal width (cm)": 2.5,
        },
    ),
    ScaleTransformation(
        transformation_options={
            "sepal length (cm)": "min_max",
            "sepal width (cm)": "robust",
        },
        quantile_range={
            "sepal width (cm)": (25.0, 75.0),
        },
    ),
    NumericalBinning(
        binning_options={
            "sepal length (cm)": "uniform",
        },
        num_bins={
            "sepal length (cm)": 5,
        },
    ),
    MathematicalOperations(
        operations_options=[
            ("sepal length (cm)", "sepal width (cm)", "add"),
        ]
    ),
]
cross = CrossTransformer(transformations)

# Fit & transform data
x_train, y_train = cross.fit_transform(x_train, y_train)
x_test, y_test = cross.transform(x_test, y_test)
```

### Import Transformations from UI

You can export transformations created in the graphical interface (UI) to a file and later import them into your scripts:

```python
import pickle
from cross import CrossTransformer

# Load transformations from file (generated through the UI)
with open("cross_transformations.pkl", "rb") as f:
    transformations = pickle.load(f)

cross = CrossTransformer(transformations)

# Apply transformations to your dataset
x_train, y_train = cross.fit_transform(x_train, y_train)
x_test, y_test = cross.transform(x_test, y_test)
```

### Save and Load Transformations

To save and reuse the transformations, save them and load them in future sessions:

```python
import pickle
from cross import CrossTransformer

# Generate transformer object
cross = CrossTransformer(transformations)

# Save transformations
transformations = cross.get_params()

with open("cross_transformations.pkl", "wb") as f:
    pickle.dump(transformations, f)

# Load transformations
with open("cross_transformations.pkl", "rb") as f:
    transformations = pickle.load(f)

cross.set_params(**transformations)
```

### Auto transformations

You can allow the library to create automatically the transformations that best fits:

```python
from cross import auto_transform, CrossTransformer
from sklearn.neighbors import KNeighborsClassifier

# Define the model
model = KNeighborsClassifier()
scoring = "accuracy"
direction = "maximize"

# Run auto transformations
transformations = auto_transform(x, y, model, scoring, direction)

# Create transformer based on transformations
transformer = CrossTransformer(transformations)

# Apply transformations to your dataset
x_train, y_train = transformer.fit_transform(x_train, y_train)
x_test, y_test = transformer.transform(x_test, y_test)
```

#### Explanation of `auto_transform`

The `auto_transform` function applies a series of data transformations to enhance the performance of a given machine learning model. It evaluates different preprocessing techniques, such as handling missing values, encoding categorical features, scaling numerical features, and more. The function iterates through various transformation strategies, selecting those that yield the best model performance based on the provided scoring metric.

**Parameters:**
- `X (np.ndarray)`: Feature matrix.
- `y (np.ndarray)`: Target variable.
- `model`: Machine learning model with a `fit` method.
- `scoring (str)`: Scoring metric for evaluation.
- `direction (str, optional)`: "maximize" to increase the score or "minimize" to decrease it (default is "maximize").
- `cv (Union[int, Callable], optional)`: Number of cross-validation folds or a custom cross-validation generator (default is 5).
- `groups (Optional[np.ndarray], optional)`: Group labels for cross-validation splitting (default is None).
- `verbose (bool, optional)`: Whether to print progress messages (default is True).

**Returns:**
- A list of applied transformations that can be used to create a `CrossTransformer` for applying them to new datasets.


## Transformations

### Clean Data

#### **Missing Values**

Handles missing values in the dataset.

- Parameters:
    - `handling_options`: Dictionary that specifies the handling strategy for each column. Options: `fill_0`, `most_frequent`, `fill_mean`, `fill_median`, `fill_mode`, `fill_knn`.
    - `n_neighbors`: Number of neighbors for K-Nearest Neighbors imputation (used with `fill_knn`).
  
```python
from cross.transformations import MissingValuesHandler

MissingValuesHandler(
    handling_options={
        'sepal width (cm)': 'fill_knn',
        'petal length (cm)': 'fill_mode',
        'petal width (cm)': 'most_frequent',
        
    },
    n_neighbors= {
        'sepal width (cm)': 5,
    }
)
```

### Numerical Features

#### **Handle Outliers**

Manages outliers in the dataset using different strategies. The action can be either cap or median, while the method can be `iqr`, `zscore`, `lof`, or `iforest`. Note that `lof` and `iforest` only accept the `median` action.

- Parameters:
    - `handling_options`: Dictionary specifying the handling strategy. The strategy is a tuple where the first element is the action (`cap` or `median`) and the second is the method (`iqr`, `zscore`, `lof`, `iforest`).
    - `thresholds`: Dictionary with thresholds for `iqr` and `zscore` methods.
    - `lof_params`: Dictionary specifying parameters for the LOF method.
    - `iforest_params`: Dictionary specifying parameters for Isolation Forest.
  
```python
from cross.transformations import OutliersHandler

OutliersHandler(
    handling_options={
        'sepal length (cm)': ('median', 'iqr'),
        'sepal width (cm)': ('cap', 'zscore'),
        'petal length (cm)': ('median', 'lof'),
        'petal width (cm)': ('median', 'iforest'),
    },
    thresholds={
        'sepal length (cm)': 1.5,
        'sepal width (cm)': 2.5,    
    },
    lof_params={
        'petal length (cm)': {
            'n_neighbors': 20,
        }
    },
    iforest_params={
        'petal width (cm)': {
            'contamination': 0.1,
        }
    }
)
```

#### **Non-Linear Transformation**

Applies non-linear transformations, including logarithmic, exponential, and Yeo-Johnson transformations.

- Parameters:
    - `transformation_options`: A dictionary specifying the transformation to be applied for each column. Options include: `log`, `exponential`, and `yeo_johnson`.

```python
from cross.transformations import NonLinearTransformation

NonLinearTransformation(
    transformation_options={
        "sepal length (cm)": "log",
        "sepal width (cm)": "exponential",
        "petal length (cm)": "yeo_johnson",
    }
)
```

#### **Quantile Transformations**

Applies quantile transformations for normalizing data.

- Parameters:
    - `transformation_options`: Dictionary specifying the transformation type. Options: `uniform`, `normal`.
  
```python
from cross.transformations import QuantileTransformation

QuantileTransformation(
    transformation_options={
        'sepal length (cm)': 'uniform',
        'sepal width (cm)': 'normal',
    }
)
```

#### **Scale Transformations**

Scales numerical data using different scaling methods.

- Parameters:
    - `transformation_options`: Dictionary specifying the scaling method for each column. Options: `min_max`, `standard`, `robust`, `max_abs`.
    -  `quantile_range`: Dictionary specifying the quantile ranges for robust scaling.
  
```python
from cross.transformations import ScaleTransformation

ScaleTransformation(
    transformation_options={
        'sepal length (cm)': 'min_max',
        'sepal width (cm)': 'standard',
        'petal length (cm)': 'robust',
        'petal width (cm)': 'max_abs',
    },
    quantile_range={
        "petal length (cm)": (25.0, 75.0),
    },
)
```

#### **Normalization**

Normalizes data using L1 or L2 norms.

- Parameters:
    - `transformation_options`: Dictionary specifying the normalization type. Options: `l1`, `l2`.
  
```python
from cross.transformations import Normalization

Normalization(
    transformation_options={
        'sepal length (cm)': 'l1',
        'sepal width (cm)': 'l2',
    }
)
```

#### **Numerical Binning**

Bins numerical columns into categories. You can now specify the column, the binning method, and the number of bins in a tuple.

- Parameters:
    - `binning_options`: List of tuples where each tuple specifies the column name, binning method, and number of bins. Options for binning methods are `uniform`, `quantile` or `kmeans`.
  
```python
from cross.transformations import NumericalBinning

NumericalBinning(
    binning_options=[
        ("sepal length (cm)", "uniform", 5),
        ("sepal width (cm)", "quantile", 6),
        ("petal length (cm)", "kmeans", 7),
    ]
)
```

#### **Mathematical Operations**

Performs mathematical operations between columns.

- Parameters:
    - `operations_options`: List of tuples specifying the columns and the operation.

- **Options**:
    - `add`: Adds the values of two columns.
    - `subtract`: Subtracts the values of two columns.
    - `multiply`: Multiplies the values of two columns.
    - `divide`: Divides the values of two columns.
    - `modulus`: Computes the modulus of two columns.
    - `hypotenuse`: Computes the hypotenuse of two columns.
    - `mean`: Calculates the mean of two columns.
  
```python
from cross.transformations import MathematicalOperations

MathematicalOperations(
    operations_options=[
        ('sepal length (cm)', 'sepal width (cm)', 'add'),
        ('petal length (cm)', 'petal width (cm)', 'subtract'),
        ('sepal length (cm)', 'petal length (cm)', 'multiply'),
        ('sepal width (cm)', 'petal width (cm)', 'divide'),
        ('sepal length (cm)', 'petal width (cm)', 'modulus'),
        ('sepal length (cm)', 'sepal width (cm)', 'hypotenuse'),
        ('petal length (cm)', 'petal width (cm)', 'mean'),
    ]
)
```

### Categorical Features

#### **Categorical Encoding**

Encodes categorical variables using various methods.

- Parameters:
    - `encodings_options`: Dictionary specifying the encoding method for each column. Options: `label`, `ordinal`, `onehot`, `dummy`, `binary`, `count`, `target`.
    - `ordinal_orders`: Specifies the order for ordinal encoding.

```python
from cross.transformations import CategoricalEncoding

CategoricalEncoding(
    encodings_options={
        'Sex': 'label',
        'Size': 'ordinal',
    },
    ordinal_orders={
        "Size": ["small", "medium", "large"]
    }
)
```

### Datetime Features

#### **Date Time Transforms**

Transforms datetime columns into useful features.

- Parameters:
    - `datetime_columns`: List of columns to extract date/time features from.
  
```python
from cross.transformations import DateTimeTransformer

DateTimeTransformer(
    datetime_columns=["date"]
)
```

### Periodic Features

#### **Cyclical Features Transforms**

Transforms cyclical features like time into a continuous representation.

- Parameters:
    - `columns_periods`: Dictionary specifying the period for each cyclical column.
  
```python
from cross.transformations import CyclicalFeaturesTransformer

CyclicalFeaturesTransformer(
    columns_periods={
        "date_minute": 60,
        "date_hour": 24,
    }
)
```

### Features Reduction

#### **Column Selection**

Allows you to select specific columns for further processing.

- Parameters:
    - `columns`: List of column names to select.
  
```python
from cross.transformations import ColumnSelection

ColumnSelection(
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
    ]
)
```


#### **Dimensionality Reduction**

Reduces the dimensionality of the dataset using various techniques, such as PCA, Factor Analysis, ICA, LDA, and others.

- Parameters:
    - `method`: The dimensionality reduction method to apply.
    - `n_components`: Number of dimensions to reduce the data to.

- **Methods**:
    - `pca`: Principal Component Analysis.
    - `factor_analysis`: Factor Analysis.
    - `ica`: Independent Component Analysis.
    - `kernel_pca`: Kernel PCA.
    - `lda`: Linear Discriminant Analysis.
    - `truncated_svd`: Truncated Singular Value Decomposition.
    - `isomap`: Isomap Embedding.
    - `lle`: Locally Linear Embedding.

- **Notes**:
For `lda`, the y target variable is required, as it uses class labels for discriminant analysis.

```python
from cross.transformations import DimensionalityReduction

DimensionalityReduction(
    method="pca",
    n_components=3
)
```
