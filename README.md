![UI Cross](assets/logo.png)

-----------------

# Cross: A Versatile Toolkit for Feature Engineering in Machine Learning

![PyPI version](https://img.shields.io/pypi/v/cross_ml)
![Downloads](https://img.shields.io/pypi/dm/cross_ml)

**Cross** is a Python library for feature engineering, providing tools for scaling, normalization, feature creation through binning, and mathematical operations between columns. It streamlines preprocessing for machine learning models, improving data quality and model performance.

## 📌 Table of Contents
- [Getting Started](#getting-started)
- [Example of Use](#example-of-use)
  - [Manual Transformations](#manual-transformations)
  - [Saving and Loading Transformations](#saving-and-loading-transformations)
  - [Automated Transformations](#automated-transformations)
- [Public Methods](#public-methods)
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


<a id="getting-started"></a>
## 🚀 Getting Started

To install **Cross**, run the following command:

```bash
pip install cross_ml
```

<a id="example-of-use"></a>
## 📖 Example of Use

<a id="manual-transformations"></a>
### 🔹 Manual Transformations

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
x_train = cross.fit_transform(x_train, y_train)
x_test = cross.transform(x_test, y_test)
```

<a id="saving-and-loading-transformations"></a>
### 💾 Saving and Loading Transformations

Save and reuse transformations for consistency across multiple sessions:

```python
import pickle
from cross import CrossTransformer

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

<a id="automated-transformations"></a>
### 🔄 Automated Transformations

Automatically select the best transformations for a dataset:

```python
from cross import auto_transform, CrossTransformer
from sklearn.neighbors import KNeighborsClassifier

# Define model
model = KNeighborsClassifier()
scoring = "accuracy"
direction = "maximize"

# Run automated feature engineering
transformations = auto_transform(x, y, model, scoring, direction)
transformer = CrossTransformer(transformations)

# Apply transformations
x_train = transformer.fit_transform(x_train, y_train)
x_test = transformer.transform(x_test, y_test)
```

---

<a id="public-methods"></a>
## 🧩 Public Methods

### **auto_transform**

Automatically applies a series of data transformations to improve model performance based on a selected scoring metric and estimator.

```python
from cross import auto_transform
```

#### **Parameters:**
- `X` (`np.ndarray`): Feature matrix.
- `y` (`np.ndarray`): Target variable.
- `model`: A machine learning model implementing a `fit` method.
- `scoring` (`str`): Evaluation metric (e.g., `"accuracy"`, `"f1"`, `"roc_auc"`).
- `direction` (`str`, optional): Optimization direction: `"maximize"` or `"minimize"`. Default is `"maximize"`.
- `cv` (`int` or callable, optional): Cross-validation strategy (e.g., number of folds or a custom splitter). Default is `None`.
- `groups` (`np.ndarray`, optional): Group labels for cross-validation splitting.
- `verbose` (`bool`, optional): Whether to display progress logs. Default is `True`.

#### **Returns:**
- `List[dict]`: A list of transformation dictionaries to be used with `CrossTransformer`.

---

### **CrossTransformer**

A wrapper for applying a pipeline of transformations defined in the Cross framework.

```python
from cross import CrossTransformer
```

#### **Constructor Parameters:**
- `transformations` (`list`, optional): List of transformation objects or dictionaries.

#### **Public Methods:**

- `fit(X, y=None)`  
    Fits each transformation in the pipeline to the dataset.
    - **Returns:** `self`

- `transform(X, y=None)`  
    Applies each fitted transformation in sequence.
    - **Returns:** Transformed feature matrix (`np.ndarray` or `pd.DataFrame`)

- `fit_transform(X, y=None)`  
    Combines `fit` and `transform` for each transformation.
    - **Returns:** Transformed feature matrix.

- `get_params(deep=True)`  
    Retrieves the parameters of the pipeline (mainly the transformations).
    - **Returns:** Dictionary of parameters.

- `set_params(**params)`  
    Sets or updates the pipeline parameters.
    - **Returns:** `self`

---

<a id="transformations"></a>
## 🔍 Transformations

<a id="missing-values-and-outliers"></a>
### 📌 Missing Values and Outliers

#### **Missing Values Indicator**

Adds indicator columns for missing values in selected features.

- Parameters:
    - `features`: List of column names to check for missing values. If None, all columns are considered.

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

Handles missing values.

- Parameters:
    - `transformation_options`: Dictionary that specifies the handling strategy for each column. Options: `fill_0`, `mean`, `median`, `most_frequent`, `knn`.
    - `n_neighbors`: Number of neighbors for K-Nearest Neighbors imputation (used with `knn`).

```python
from cross.transformations import MissingValuesHandler

MissingValuesHandler(
    transformation_options={
        'sepal width (cm)': 'knn',
        'petal length (cm)': 'mean',
        'petal width (cm)': 'most_frequent',
        
    },
    n_neighbors= {
        'sepal width (cm)': 5,
    }
)
```

#### **Handle Outliers**

Detects and mitigates outliers using methods like `iqr`, `zscore`, `lof`, or `iforest`.

- Parameters:
    - `transformation_options`: Dictionary specifying the handling strategy. The strategy is a tuple where the first element is the action (`cap` or `median`) and the second is the method (`iqr`, `zscore`, `lof`, `iforest`).
    - `thresholds`: Dictionary with thresholds for `iqr` and `zscore` methods.
    - `lof_params`: Dictionary specifying parameters for the LOF method.
    - `iforest_params`: Dictionary specifying parameters for Isolation Forest.

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

<a id="data-distribution-and-scaling"></a>
### 📌 Data Distribution and Scaling

#### **Non-Linear Transformation**

Applies logarithmic, exponential, or Yeo-Johnson transformations.

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

Transforms data to follow a normal or uniform distribution.

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

<a id="numerical-features"></a>
### 📌 Numerical Features

#### **Spline Transformations**

Applies Spline transformation to numerical features.

- Parameters:
    - `transformation_options`: Dictionary specifying the spline transformation settings for each column. Options include different numbers of knots and degrees.

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

<a id="categorical-features"></a>
### 📌 Categorical Features

#### **Categorical Encoding**

Encodes categorical variables using various methods.

- Parameters:
    - `encodings_options`: Dictionary specifying the encoding method for each column.
    - `ordinal_orders`: Specifies the order for ordinal encoding.

- **Encodings**:
    - `backward_diff`: Uses backward difference coding to compare each category to the previous one.
    - `basen`: Encodes categorical features using a base-N representation.
    - `binary`: Converts categorical variables into binary representations.
    - `catboost`: Implements the CatBoost encoding, which is a target-based encoding method.
    - `count`: Replaces categories with the count of occurrences in the dataset.
    - `dummy`: Applies dummy coding, similar to one-hot encoding but with one less category to avoid collinearity.
    - `glmm`: Uses Generalized Linear Mixed Models to encode categorical variables.
    - `gray`: Converts categories into Gray code, a binary numeral system where two successive values differ in only one bit.
    - `hashing`: Uses a hashing trick to encode categorical features into a fixed number of dimensions.
    - `helmert`: Compares each level of a categorical variable to the mean of subsequent levels.
    - `james_stein`: Applies James-Stein shrinkage estimation for target encoding.
    - `label`: Assigns each category a unique integer label.
    - `loo`: Uses leave-one-out target encoding to replace categories with the mean target value, excluding the current row.
    - `m_estimate`: A variant of target encoding that applies an m-estimate to regularize values.
    - `onehot`: Converts categorical variables into binary vectors where each category is represented by a separate column.
    - `ordinal`: Replaces categories with ordinal values based on their ordering.
    - `polynomial`: Applies polynomial contrast coding to categorical variables.
    - `quantile`: Maps categorical variables to quantiles based on their distribution.
    - `rankhot`: Encodes categories based on their ranking, similar to one-hot but considering order.
    - `sum`: Uses sum coding to compare each level to the overall mean.
    - `target`: Encodes categories using the mean of the target variable for each category.
    - `woe`: Applies Weight of Evidence (WoE) encoding, useful in logistic regression by transforming categorical data into log odds.

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

<a id="periodic-features"></a>
### 📌 Periodic Features

#### **Date Time Transforms**

Transforms datetime columns into useful features.

- Parameters:
    - `features`: List of columns to extract date/time features from. If None, all datetime columns are considered.

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

```python
from cross.transformations import CyclicalFeaturesTransformer

CyclicalFeaturesTransformer(
    transformation_options={
        "date_minute": 60,
        "date_hour": 24,
    }
)
```

<a id="features-reduction"></a>
### 📌 Features Reduction

#### **Column Selection**

Allows you to select specific columns for further processing.

- Parameters:
    - `features`: List of column names to select.

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

```python
from cross.transformations import DimensionalityReduction

DimensionalityReduction(
    method="pca",
    n_components=3
)
```

---

## 🛠️ Contributing
We welcome contributions! Feel free to submit pull requests or report issues.

## 📄 License
Cross is open-source and licensed under the MIT License.

---

🚀 **Enhance your feature engineering pipeline with Cross!**
