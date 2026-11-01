//! The error type shared by the models, preprocessing, and metrics.

use std::error::Error;
use std::fmt;

/// Everything that can go wrong with bad input.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ModelError {
    /// No data was provided.
    EmptyData,
    /// Rows have no features.
    NoFeatures,
    /// Two inputs that must be the same length are not.
    LengthMismatch { left: usize, right: usize },
    /// A row has a different number of features from the first row.
    RaggedRow {
        row: usize,
        expected: usize,
        found: usize,
    },
    /// A row contains `NaN` or infinity.
    NonFiniteValue { row: usize },
    /// A classification label is not 0 or 1.
    InvalidLabel { row: usize },
    /// Classification training data contains only one class.
    SingleClass,
    /// Values are too large to compute with.
    NumericOverflow,
    /// A [`TrainConfig`](crate::training::TrainConfig) setting is out of range.
    InvalidConfig(&'static str),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyData => write!(f, "no data was provided"),
            Self::NoFeatures => write!(f, "rows have no features"),
            Self::LengthMismatch { left, right } => {
                write!(f, "lengths do not match: {} and {}", left, right)
            }
            Self::RaggedRow {
                row,
                expected,
                found,
            } => {
                write!(
                    f,
                    "row {} has {} features, expected {}",
                    row, found, expected
                )
            }
            Self::NonFiniteValue { row } => write!(f, "row {} contains NaN or infinity", row),
            Self::InvalidLabel { row } => write!(f, "row {} has a label that is not 0 or 1", row),
            Self::SingleClass => write!(f, "training data must contain both classes"),
            Self::NumericOverflow => write!(f, "values are too large to compute with"),
            Self::InvalidConfig(reason) => write!(f, "invalid config: {}", reason),
        }
    }
}

impl Error for ModelError {}
