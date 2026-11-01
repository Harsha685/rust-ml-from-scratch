//! Settings and results shared by every model trained with gradient descent.

use crate::error::ModelError;

/// Training settings.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// The learning rate. `None` lets the model choose one. Either way, the model lowers
    /// a rate that is too high, so training can't diverge. The rate actually used is in
    /// [`FitReport::learning_rate`].
    pub learning_rate: Option<f64>,
    /// The most gradient descent steps to take.
    pub max_iterations: usize,
    /// Stop when the size of the gradient, measured on scaled data, is this small.
    pub tolerance: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: None,
            max_iterations: 1_000_000,
            tolerance: 1e-8,
        }
    }
}

/// Why training stopped.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum StopReason {
    /// The gradient reached zero, so this is the best possible model.
    Converged,
    /// Classification only. A line splits the training data perfectly, so the loss can approach
    /// 0 but never reach it, and the weights would grow forever. Training stops once the loss
    /// is tiny, and the weights have no single correct value.
    Separated,
    /// Training hit [`TrainConfig::max_iterations`] first, so results may be inexact.
    IterationLimit,
}

/// What happened during training.
#[derive(Debug, Clone)]
pub struct FitReport {
    /// How many gradient descent steps were taken.
    pub iterations: usize,
    /// Why training stopped.
    pub stop_reason: StopReason,
    /// The learning rate in use when training stopped.
    pub learning_rate: f64,
    /// The model's loss on the training data: mean squared error for
    /// regression, log loss for classification.
    pub train_loss: f64,
}

pub(crate) fn validate_config(config: &TrainConfig) -> Result<(), ModelError> {
    if let Some(rate) = config.learning_rate
        && (!rate.is_finite() || rate <= 0.0)
    {
        return Err(ModelError::InvalidConfig("learning_rate must be above 0"));
    }
    if config.max_iterations == 0 {
        return Err(ModelError::InvalidConfig("max_iterations must be above 0"));
    }
    if !config.tolerance.is_finite() || config.tolerance < 0.0 {
        return Err(ModelError::InvalidConfig("tolerance must not be negative"));
    }
    Ok(())
}

// CHECKS THE SHAPE OF THE DATA AND THAT EVERY FEATURE IS A REAL NUMBER
pub(crate) fn validate_rows(xs: &[Vec<f64>], ys: &[f64]) -> Result<(), ModelError> {
    if xs.is_empty() {
        return Err(ModelError::EmptyData);
    }
    if xs.len() != ys.len() {
        return Err(ModelError::LengthMismatch {
            left: xs.len(),
            right: ys.len(),
        });
    }
    let n_features = xs[0].len();
    if n_features == 0 {
        return Err(ModelError::NoFeatures);
    }
    for (row, x) in xs.iter().enumerate() {
        if x.len() != n_features {
            return Err(ModelError::RaggedRow {
                row,
                expected: n_features,
                found: x.len(),
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::NonFiniteValue { row });
        }
    }
    Ok(())
}

pub(crate) fn validate_labels(ys: &[f64]) -> Result<(), ModelError> {
    for (row, y) in ys.iter().enumerate() {
        if *y != 0.0 && *y != 1.0 {
            return Err(ModelError::InvalidLabel { row });
        }
    }
    Ok(())
}
