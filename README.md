![UI Cross](assets/logo.png)

-----------------

# Cross: a versatile toolkit for data preprocessing and feature engineering in machine learning

Cross is a Python library for data processing to train machine learning models, featuring scaling, normalization, feature creation through binning, and various mathematical operations between columns. It includes a graphical interface for exploring and generating transformations, with the option to export and reuse them.

![UI Cross](assets/ui_outliers_handling.png)


## Getting Started

To install the Cross library, run the following command:

```bash
pip install cross
```

You can initialize the graphical interface with the command:

```bash
cross run
```

Access the application by navigating to `http://localhost:8501` in your browser.


## Example of Use

### Define transformations

```python
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
            "sepal width (cm)": "min_max",
        }
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

# Load transformations from file
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

# Save transformations
transformations = cross.get_params()

with open("cross_transformations.pkl", "wb") as f:
    pickle.dump(transformations, f)

# Load transformations
with open("cross_transformations.pkl", "rb") as f:
    transformations = pickle.load(f)

cross.set_params(**transformations)
```

## Transformations

- [Clean Data](#clean-data)
    - [Column Selection](#column-selection)
    - [Column Casting](#column-casting)
    - [Missing Values](#missing-values)
    - [Handle Outliers](#handle-outliers)
- [Preprocessing](#preprocessing)
    - [Non-Linear Transformation](#non-linear-transformation)
    - [Quantile Transformations](#quantile-transformations)
    - [Scale Transformations](#scale-transformations)
    - [Normalization](#normalization)
- [Feature Engineering](#feature-engineering)
    - [Categorical Encoding](#categorical-encoding)
    - [Date Time Transforms](#date-time-transforms)
    - [Cyclical Features Transforms](#cyclical-features-transforms)
    - [Numerical Binning](#numerical-binning)
    - [Mathematical Operations](#mathematical-operations)

### Clean Data

#### **Column Selection**

Allows you to select specific columns for further processing.

- Parameters:
    - `columns`: List of column names to select.
  
```python
ColumnSelection(
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
    ]
)
```

#### **Column Casting**

Casts columns to specific data types.

- Parameters:
    - `cast_options`: A dictionary specifying the type for each column. Options include: `category`, `number`, `bool`, `datetime`, `timedelta`.
  
```python
CastColumns(
    cast_options={
        "sepal length (cm)": "number",
        "sepal width (cm)": "number",
    }
)
```

#### **Missing Values**

Handles missing values in the dataset.

- Parameters:
    - `handling_options`: Dictionary that specifies the handling strategy for each column. Options: `fill_0`, `most_frequent`, `fill_mean`, `fill_median`, `fill_mode`, `fill_knn`.
    - `n_neighbors`: Number of neighbors for K-Nearest Neighbors imputation (used with `fill_knn`).
  
```python
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

#### **Handle Outliers**

Manages outliers in the dataset using different strategies. The action can be either cap or median, while the method can be `iqr`, `zscore`, `lof`, or `iforest`. Note that `lof` and `iforest` only accept the `median` action.

- Parameters:
    - `handling_options`: Dictionary specifying the handling strategy. The strategy is a tuple where the first element is the action (`cap` or `median`) and the second is the method (`iqr`, `zscore`, `lof`, `iforest`).
    - `thresholds`: Dictionary with thresholds for `iqr` and `zscore` methods.
    - `lof_params`: Dictionary specifying parameters for the LOF method.
    - `iforest_params`: Dictionary specifying parameters for Isolation Forest.
  
```python
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

### Preprocessing

### **Non-Linear Transformation**

Applies non-linear transformations, including logarithmic, exponential, and Yeo-Johnson transformations.

- Parameters:
    - `transformation_options`: A dictionary specifying the transformation to be applied for each column. Options include: `log`, `exponential`, and `yeo_johnson`.

```python
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
  
```python
ScaleTransformation(
    transformation_options={
        'sepal length (cm)': 'min_max',
        'sepal width (cm)': 'standard',
        'petal length (cm)': 'robust',
        'petal width (cm)': 'max_abs',
    }
)
```

#### **Normalization**

Normalizes data using L1 or L2 norms.

- Parameters:
    - `transformation_options`: Dictionary specifying the normalization type. Options: `l1`, `l2`.
  
```python
Normalization(
    transformation_options={
        'sepal length (cm)': 'l1',
        'sepal width (cm)': 'l2',
    }
)
```

### Feature Engineering

#### **Categorical Encoding**

Encodes categorical variables using various methods.

- Parameters:
    - `encodings_options`: Dictionary specifying the encoding method for each column. Options: `label`, `ordinal`, `onehot`, `dummy`, `binary`, `count`, `target`.
    - `ordinal_orders`: Specifies the order for ordinal encoding.

```python
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

#### **Date Time Transforms**

Transforms datetime columns into useful features.

- Parameters:
    - `datetime_columns`: List of columns to extract date/time features from.
  
```python
DateTimeTransformer(
    datetime_columns=["date"]
)
```

#### **Cyclical Features Transforms**

Transforms cyclical features like time into a continuous representation.

- Parameters:
    - `columns_periods`: Dictionary specifying the period for each cyclical column.
  
```python
CyclicalFeaturesTransformer(
    columns_periods={
        "date_minute": 60,
        "date_hour": 24,
    }
)
```

#### **Numerical Binning**

Bins numerical columns into categories. You can now specify the column, the binning method, and the number of bins in a tuple.

- Parameters:
    - `binning_options`: List of tuples where each tuple specifies the column name, binning method, and number of bins. Options for binning methods are `uniform`, `quantile` or `kmeans`.
  
```python
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
