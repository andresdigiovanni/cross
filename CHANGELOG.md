# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2025-03-21

### Added

- Extrapolation parameter in spline transformation
- Filter not used transformations after auto_transform
- Subsample data for auto_transform

### Changed

- Rename parameters to standardize

## [1.2.1] - 2025-03-11

### Changed

- Order in column selection and dimensionality reduction

### Fixed

- Initialize transformations

## [1.2.0] - 2025-03-10

### Added

- Quantile range parameter in RobustScaler
- Spline transformation for numerical data
- Missing indicator column for missing values
- Backward Difference encoding for categorical data
- Base N encoding for categorical data
- Cat Boost encoding for categorical data
- GLMM encoding for categorical data
- Gray encoding for categorical data
- Hashing encoding for categorical data
- Helmert encoding for categorical data
- JamesStein encoding for categorical data
- Leave One Out enconding for categorical data
- Ordinal encoding for categorical data
- MEstimate encoding for categorical data
- Polynomial encoding for categorical data
- Quantile encoding for categorical data
- Rank Hot encoding for categorical data
- Sum encoding for categorical data
- WOE encoding for categorical data

### Removed

- User Interface

## [1.1.1] - 2025-01-30

### Fixed

- Wrong package

## [1.1.0] - 2025-01-30

### Added

- CV and Groups parameters to auto_transform function
- Normalization and Quantile transformations in auto_transform
- Dimensionality Reduction transformations

### Changed

- Internal use of RecursiveFeatureAddition

### Fixed

- Target encoding for multiclass in CategoricalEncoding
- Handle models that not support nan values

## [1.0.0] - 2025-01-03

### Fixed

- Improve performance selecting features using probe method
- Fixed OutliersHandler: store handler objects

### Changed

- Changed namespaces

### Removed

- CorrelatedSubstringEncoder transformer

## [0.2.0] - 2024-11-19

### Added

- CorrelatedSubstringEncoder transformer for text data

### Fixed

- Ignore unknown values in CategoricalEncoding

### Changed

- Changed parameter PowerTransformer standardize to False in NonLinearTransformation

## [0.1.0] - 2024-10-25

### Added

- Initial version
