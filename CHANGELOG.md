# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CV and Groups parameters to auto_transform function
- Normalization and Quantile transformations in auto_transform

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
