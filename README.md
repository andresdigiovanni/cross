![UI Cross](assets/logo.png)

-----------------

# Cross: a versatile toolkit for feature engineering in machine learning

![PyPI version](https://img.shields.io/pypi/v/cross_ml)
![Downloads](https://img.shields.io/pypi/dm/cross_ml)

Cross is a Python library for feature engineering to train machine learning models, featuring scaling, normalization, feature creation through binning, and various mathematical operations between columns.

- [Getting Started](#getting-started)
- [Example of Use](#example-of-use)
  - [Define transformations](#define-transformations)
  - [Save and Load Transformations](#save-and-load-transformations)
  - [Auto transformations](#auto-transformations)
- [Transformations](#transformations)
  - [Missing Values and Outliers](#missing-values-and-outliers)
    - [Missing Values Indicator](#missing-values-indicator)
    - [Missing Values Handler](#missing-values-handler)
    - [Handle Outliers](#handle-outliers)
  - [Data Distribution and Scaling](#data-distribution-and-scaling)
    - [Non-Linear Transformation](#non-linear-transformation)
    - [Quantile Transformations](#quantile-transformations)
    - [Scale Transformations](#scale-transformations)
    - [Normalization](#normalization)
  - [Numerical Features](#numerical-features)
    - [Spline Transformations](#spline-transformations)
    - [Numerical Binning](#numerical-binning)
    - [Mathematical Operations](#mathematical-operations)
  - [Categorical Features](#categorical-features)
    - [Categorical Encoding](#categorical-encoding)
  - [Periodic Features](#periodic-features)
    - [Date Time Transforms](#date-time-transforms)
    - [Cyclical Features Transforms](#cyclical-features-transforms)
  - [Features Reduction](#features-reduction)
    - [Column Selection](#column-selection)
    - [Dimensionality Reduction](#dimensionality-reduction)


## Getting Started

To install the Cross library, run the following command:

```bash
pip install cross_ml
```

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
        transformation_options={
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
        transformation_options={
            "sepal length (cm)": ("uniform", 5),
        }
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
- `distribution_similarity_threshold (float, optional)`: Significance level to accept that distributions are similar. Used to select a representative subset of the data. Defaults to 0.05.
- `verbose (bool, optional)`: Whether to print progress messages (default is True).

**Returns:**
- A list of applied transformations that can be used to create a `CrossTransformer` for applying them to new datasets.


## Transformations

### Missing Values and Outliers

#### **Missing Values Indicator**

Detects and encodes missing values in the dataset by adding indicator columns.

- Parameters:
    - `features`: List of column names to check for missing values. If None, all columns are considered.

- Example Usage:

```python
from cross.transformations import MissingValuesIndicator

MissingValuesIndicator(
    features=[
        'sepal width (cm)',
        'petal length (cm)',
    ]
)
```

#### **Missing Values Handler**

Handles missing values in the dataset.

- Parameters:
    - `transformation_options`: Dictionary that specifies the handling strategy for each column. Options: `fill_0`, `most_frequent`, `fill_mean`, `fill_median`, `fill_mode`, `fill_knn`.
    - `n_neighbors`: Number of neighbors for K-Nearest Neighbors imputation (used with `fill_knn`).

- Example Usage:

```python
from cross.transformations import MissingValuesHandler

MissingValuesHandler(
    transformation_options={
        'sepal width (cm)': 'fill_knn',
        'petal length (cm)': 'fill_mode',
        'petal width (cm)': 'most_frequent',
        
    },
    n_neighbors= {
        'sepal width (cm)': 5,
    }
)
```

#### **Handle Outliers**

Manages outliers in the dataset using different strategies. The action can be either cap or median, while the method can be `iqr`, `zscore`, `lof`, or `iforest`. Note that `lof` and `iforest` only accept the `median` action.

- Parameters:
    - `transformation_options`: Dictionary specifying the handling strategy. The strategy is a tuple where the first element is the action (`cap` or `median`) and the second is the method (`iqr`, `zscore`, `lof`, `iforest`).
    - `thresholds`: Dictionary with thresholds for `iqr` and `zscore` methods.
    - `lof_params`: Dictionary specifying parameters for the LOF method.
    - `iforest_params`: Dictionary specifying parameters for Isolation Forest.

- Example Usage:

```python
from cross.transformations import OutliersHandler

OutliersHandler(
    transformation_options={
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

### Data Distribution and Scaling

#### **Non-Linear Transformation**

Applies non-linear transformations, including logarithmic, exponential, and Yeo-Johnson transformations.

- Parameters:
    - `transformation_options`: A dictionary specifying the transformation to be applied for each column. Options include: `log`, `exponential`, and `yeo_johnson`.

- Example Usage:

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

- Example Usage:

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

- Example Usage:

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

- Example Usage:

```python
from cross.transformations import Normalization

Normalization(
    transformation_options={
        'sepal length (cm)': 'l1',
        'sepal width (cm)': 'l2',
    }
)
```

### Numerical Features

#### **Spline Transformations**

Applies Spline transformation to numerical features.

- Parameters:
    - `transformation_options`: Dictionary specifying the spline transformation settings for each column. Options include different numbers of knots and degrees.

- Example Usage:

```python
from cross.transformations import SplineTransformation

SplineTransformation(
    transformation_options={
        'sepal length (cm)': {'degree': 3, 'n_knots': 3},
        'sepal width (cm)': {'degree': 3, 'n_knots': 5},
    }
)
```


#### **Numerical Binning**

Bins numerical columns into categories. You can now specify the column, the binning method, and the number of bins in a tuple.

- Parameters:
    - `transformation_options`: Dictionary specifying the binning method and number of bins for each column. Options for binning methods are `uniform`, `quantile` or `kmeans`.

- Example Usage:

```python
from cross.transformations import NumericalBinning

NumericalBinning(
    transformation_options={
        "sepal length (cm)": ("uniform", 5),
        "sepal width (cm)": ("quantile", 6),
        "petal length (cm)": ("kmeans", 7),
    }
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

- Example Usage:

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
    - `encodings_options`: Dictionary specifying the encoding method for each column.
    - `ordinal_orders`: Specifies the order for ordinal encoding.

- **Encodings**:
    - backward_diff
    - basen
    - binary
    - catboost
    - count
    - dummy
    - glmm
    - gray
    - hashing
    - helmert
    - james_stein
    - label
    - loo
    - m_estimate
    - onehot
    - ordinal
    - polynomial
    - quantile
    - rankhot
    - sum
    - target
    - woe

- Example Usage:

```python
from cross.transformations import CategoricalEncoding

CategoricalEncoding(
    transformation_options={
        'Sex': 'label',
        'Size': 'ordinal',
    },
    ordinal_orders={
        "Size": ["small", "medium", "large"]
    }
)
```

### Periodic Features

#### **Date Time Transforms**

Transforms datetime columns into useful features.

- Parameters:
    - `features`: List of columns to extract date/time features from. If None, all datetime columns are considered.

- Example Usage:

```python
from cross.transformations import DateTimeTransformer

DateTimeTransformer(
    features=["date"]
)
```

#### **Cyclical Features Transforms**

Transforms cyclical features like time into a continuous representation.

- Parameters:
    - `transformation_options`: Dictionary specifying the period for each cyclical column.

- Example Usage:

```python
from cross.transformations import CyclicalFeaturesTransformer

CyclicalFeaturesTransformer(
    transformation_options={
        "date_minute": 60,
        "date_hour": 24,
    }
)
```

### Features Reduction

#### **Column Selection**

Allows you to select specific columns for further processing.

- Parameters:
    - `features`: List of column names to select.

- Example Usage:

```python
from cross.transformations import ColumnSelection

ColumnSelection(
    features=[
        "sepal length (cm)",
        "sepal width (cm)",
    ]
)
```

#### **Dimensionality Reduction**

Reduces the dimensionality of the dataset using various techniques, such as PCA, Factor Analysis, ICA, LDA, and others.

- Parameters:
    - `features`: List of column names to apply the dimensionality reduction. If None, all columns are considered.
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

- Example Usage:

```python
from cross.transformations import DimensionalityReduction

DimensionalityReduction(
    method="pca",
    n_components=3
)
```
