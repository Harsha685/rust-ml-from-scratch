//! Scores for regression and classification models.

use crate::error::ModelError;
use crate::training::validate_labels;

fn validate_lengths(left: usize, targets: &[f64]) -> Result<(), ModelError> {
    if left != targets.len() {
        return Err(ModelError::LengthMismatch {
            left,
            right: targets.len(),
        });
    }
    if targets.is_empty() {
        return Err(ModelError::EmptyData);
    }
    Ok(())
}

/// Mean squared error.
pub fn mse(predictions: &[f64], targets: &[f64]) -> Result<f64, ModelError> {
    validate_lengths(predictions.len(), targets)?;
    let mut sum = 0.0;
    for (prediction, target) in predictions.iter().zip(targets.iter()) {
        let error = prediction - target;
        sum += error * error;
    }
    Ok(sum / predictions.len() as f64)
}

/// R²: 1 is perfect, and 0 is no better than always guessing the mean of the targets.
/// A constant target scores 1 if predicted perfectly, otherwise 0.
pub fn r2(predictions: &[f64], targets: &[f64]) -> Result<f64, ModelError> {
    validate_lengths(predictions.len(), targets)?;

    let ss_res = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum::<f64>();

    let mean = targets.iter().sum::<f64>() / targets.len() as f64;
    let ss_tot = targets.iter().map(|t| (t - mean).powi(2)).sum::<f64>();

    if ss_tot == 0.0 {
        return Ok(if ss_res == 0.0 { 1.0 } else { 0.0 });
    }
    Ok(1.0 - ss_res / ss_tot)
}

/// Log loss, also called binary cross-entropy. Targets must be 0 or 1. Probabilities
/// are clamped to just inside 0 and 1, because ln(0) is negative infinity.
pub fn log_loss(probabilities: &[f64], targets: &[f64]) -> Result<f64, ModelError> {
    validate_lengths(probabilities.len(), targets)?;
    validate_labels(targets)?;
    let mut total = 0.0;

    for (row, (p, y)) in probabilities.iter().zip(targets.iter()).enumerate() {
        if !p.is_finite() {
            return Err(ModelError::NonFiniteValue { row });
        }
        let p = p.clamp(1e-15, 1.0 - 1e-15);
        total += y * p.ln() + (1.0 - y) * (1.0 - p).ln();
    }

    Ok(-total / probabilities.len() as f64)
}

/// The fraction of predictions that are correct. Targets must be 0 or 1.
pub fn accuracy(predictions: &[u8], targets: &[f64]) -> Result<f64, ModelError> {
    validate_lengths(predictions.len(), targets)?;
    validate_labels(targets)?;

    let mut correct = 0;
    for (p, y) in predictions.iter().zip(targets.iter()) {
        if *p as f64 == *y {
            correct += 1;
        }
    }

    Ok(correct as f64 / predictions.len() as f64)
}

/// Of everything predicted 1, the fraction that really was 1.
/// Is 0 if nothing was predicted 1. Targets must be 0 or 1.
pub fn precision(predictions: &[u8], targets: &[f64]) -> Result<f64, ModelError> {
    validate_lengths(predictions.len(), targets)?;
    validate_labels(targets)?;

    let mut true_positives = 0;
    let mut false_positives = 0;
    for (p, y) in predictions.iter().zip(targets.iter()) {
        if *p == 1 && *y == 1.0 {
            true_positives += 1;
        }
        if *p == 1 && *y == 0.0 {
            false_positives += 1;
        }
    }

    let predicted_ones = true_positives + false_positives;
    if predicted_ones == 0 {
        return Ok(0.0);
    }
    Ok(true_positives as f64 / predicted_ones as f64)
}

/// Of everything that really was 1, the fraction that was caught.
/// Is 0 if there were no 1s. Targets must be 0 or 1.
pub fn recall(predictions: &[u8], targets: &[f64]) -> Result<f64, ModelError> {
    validate_lengths(predictions.len(), targets)?;
    validate_labels(targets)?;

    let mut true_positives = 0;
    let mut false_negatives = 0;
    for (p, y) in predictions.iter().zip(targets.iter()) {
        if *p == 1 && *y == 1.0 {
            true_positives += 1;
        }
        if *p == 0 && *y == 1.0 {
            false_negatives += 1;
        }
    }

    let actual_ones = true_positives + false_negatives;
    if actual_ones == 0 {
        return Ok(0.0);
    }
    Ok(true_positives as f64 / actual_ones as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance
    }

    #[test]
    fn regression_score_edge_cases() {
        assert_eq!(mse(&[1.0, 3.0], &[1.0, 1.0]).unwrap(), 2.0);
        assert_eq!(r2(&[1.0, 2.0], &[1.0, 2.0]).unwrap(), 1.0);
        assert_eq!(r2(&[1.5, 1.5], &[1.0, 2.0]).unwrap(), 0.0);
        assert_eq!(r2(&[5.0, 5.0], &[5.0, 5.0]).unwrap(), 1.0);
        assert_eq!(r2(&[4.0, 6.0], &[5.0, 5.0]).unwrap(), 0.0);
        assert_eq!(mse(&[], &[]).unwrap_err(), ModelError::EmptyData);
        assert!(matches!(
            r2(&[1.0], &[]),
            Err(ModelError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn log_loss_values() {
        let loss = |p: &[f64], y: &[f64]| log_loss(p, y).unwrap();
        assert!(close(loss(&[0.9, 0.2], &[1.0, 0.0]), 0.1643, 1e-4));
        assert!(close(
            loss(&[0.5, 0.5], &[1.0, 0.0]),
            std::f64::consts::LN_2,
            1e-12
        ));
        assert!(close(
            loss(&[0.1, 0.9], &[1.0, 0.0]),
            std::f64::consts::LN_10,
            1e-12
        ));
    }

    #[test]
    fn log_loss_stays_finite_at_the_extremes() {
        let loss = |p: &[f64], y: &[f64]| log_loss(p, y).unwrap();
        assert!(close(loss(&[1.0], &[0.0]), 34.54, 0.01));
        assert!(close(loss(&[1.0], &[1.0]), 0.0, 1e-12));
        assert!(close(loss(&[0.0], &[0.0]), 0.0, 1e-12));
    }

    #[test]
    fn scores() {
        let actual = [1.0, 0.0, 0.0, 0.0];
        let eager = [1, 1, 1, 0];
        let cautious = [0, 0, 0, 0];
        assert_eq!(accuracy(&eager, &actual).unwrap(), 0.5);
        assert!(close(precision(&eager, &actual).unwrap(), 1.0 / 3.0, 1e-12));
        assert_eq!(recall(&eager, &actual).unwrap(), 1.0);
        assert_eq!(accuracy(&cautious, &actual).unwrap(), 0.75);
        assert_eq!(precision(&cautious, &actual).unwrap(), 0.0);
        assert_eq!(recall(&cautious, &actual).unwrap(), 0.0);
    }

    #[test]
    fn scoring_rejects_bad_input() {
        assert_eq!(accuracy(&[], &[]), Err(ModelError::EmptyData));
        assert!(matches!(
            precision(&[1], &[]),
            Err(ModelError::LengthMismatch { .. })
        ));
        assert_eq!(
            recall(&[1], &[2.0]),
            Err(ModelError::InvalidLabel { row: 0 })
        );
        assert_eq!(
            log_loss(&[f64::NAN], &[1.0]),
            Err(ModelError::NonFiniteValue { row: 0 })
        );
    }
}
