//! Logistic regression trained with gradient descent, for predicting one of two classes.

use crate::error::ModelError;
use crate::metrics::log_loss;
use crate::preprocessing::standardize;
use crate::training::{
    FitReport, StopReason, TrainConfig, validate_config, validate_labels, validate_rows,
};

/// A trained logistic regression model. Weights and bias are in the original
/// units of the data, so new inputs never need scaling before predicting.
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LogisticRegression {
    /// Trains a model by scaling the features, running gradient descent, and
    /// converting the result back to original units.
    ///
    /// `xs` holds one row of features per example, and `ys` holds their labels, which must be
    /// exactly 0 or 1. Returns an error for empty or malformed data, `NaN` or infinite values,
    /// other labels, training data with only one class, or invalid settings.
    pub fn fit(
        xs: &[Vec<f64>],
        ys: &[f64],
        config: &TrainConfig,
    ) -> Result<(Self, FitReport), ModelError> {
        validate_data(xs, ys)?;
        validate_config(config)?;

        let (scaled_xs, means, stds) = standardize(xs)?;
        let mut params = Params {
            weights: vec![0.0; means.len()],
            bias: 0.0,
        };
        let (iterations, stop_reason, learning_rate) =
            gradient_descent(&mut params, &scaled_xs, ys, config)?;

        // UNDO THE SCALING: z = b + Σ w · (x − mean) / std
        let mut weights = vec![0.0; means.len()];
        let mut bias = params.bias;
        for j in 0..weights.len() {
            if stds[j] != 0.0 {
                weights[j] = params.weights[j] / stds[j];
                bias -= weights[j] * means[j];
            }
        }
        if !bias.is_finite() || weights.iter().any(|w| !w.is_finite()) {
            return Err(ModelError::NumericOverflow);
        }

        let model = Self { weights, bias };
        let train_loss = log_loss(&model.predict_proba_all(xs)?, ys)?;
        let report = FitReport {
            iterations,
            stop_reason,
            learning_rate,
            train_loss,
        };
        Ok((model, report))
    }

    /// Calculates the probability of class 1 for one row of features.
    pub fn predict_proba(&self, x: &[f64]) -> Result<f64, ModelError> {
        if x.len() != self.weights.len() {
            return Err(ModelError::LengthMismatch {
                left: x.len(),
                right: self.weights.len(),
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::NonFiniteValue { row: 0 });
        }
        let z = evaluate(&self.weights, self.bias, x);
        // INFINITY IS FINE, THE SIGMOID TURNS IT INTO 0 OR 1. NaN MEANS +inf AND -inf MET.
        if z.is_nan() {
            return Err(ModelError::NumericOverflow);
        }
        Ok(sigmoid(z))
    }

    /// Predicts the class, 0 or 1, for one row of features.
    pub fn predict(&self, x: &[f64]) -> Result<u8, ModelError> {
        Ok(if self.predict_proba(x)? >= 0.5 { 1 } else { 0 })
    }

    /// Calculates the probability of class 1 for every row.
    pub fn predict_proba_all(&self, xs: &[Vec<f64>]) -> Result<Vec<f64>, ModelError> {
        xs.iter().map(|x| self.predict_proba(x)).collect()
    }

    /// Predicts the class, 0 or 1, for every row.
    pub fn predict_all(&self, xs: &[Vec<f64>]) -> Result<Vec<u8>, ModelError> {
        xs.iter().map(|x| self.predict(x)).collect()
    }
}

// WEIGHTS AND BIAS WHILE TRAINING, IN SCALED UNITS
struct Params {
    weights: Vec<f64>,
    bias: f64,
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

// w·x + b
fn evaluate(weights: &[f64], bias: f64, x: &[f64]) -> f64 {
    let mut total = bias;
    for (u, v) in weights.iter().zip(x.iter()) {
        total += u * v;
    }
    total
}

//CALCULATES GRADIENTS
fn gradients(params: &Params, xs: &[Vec<f64>], ys: &[f64]) -> (Vec<f64>, f64) {
    // return (gradient of w, gradient of b)
    let mut grad_w = vec![0.0; params.weights.len()];
    let mut grad_b = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let error = sigmoid(evaluate(&params.weights, params.bias, x)) - y;
        for j in 0..x.len() {
            grad_w[j] += error * x[j];
        }
        grad_b += error;
    }
    let n = xs.len() as f64;
    for g in grad_w.iter_mut() {
        *g /= n;
    }
    (grad_w, grad_b / n)
}

// THE LARGEST LEARNING RATE THAT IS GUARANTEED TO LOWER THE LOSS ON EVERY STEP. THE SIGMOID'S
// SLOPE IS AT MOST 1/4, WHICH LIMITS HOW FAST THE GRADIENT OF LOG LOSS CAN CHANGE TO
//     L = (mean of |x|² + 1) / 4
// AND GRADIENT DESCENT ALWAYS GOES DOWNHILL WHEN THE LEARNING RATE IS AT MOST 1 / L.
fn safe_learning_rate(xs: &[Vec<f64>]) -> f64 {
    let total: f64 = xs
        .iter()
        .map(|x| x.iter().map(|v| v * v).sum::<f64>())
        .sum();
    4.0 / (total / xs.len() as f64 + 1.0)
}

// TRUE WHEN EVERY ROW IS ON THE CORRECT SIDE OF THE LINE AND THE LOSS IS TINY
fn is_separated(params: &Params, xs: &[Vec<f64>], ys: &[f64]) -> bool {
    const SEPARATED_LOSS: f64 = 1e-4;

    let mut total = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let p = sigmoid(evaluate(&params.weights, params.bias, x));
        if (p >= 0.5) != (*y == 1.0) {
            return false;
        }
        total -= if *y == 1.0 {
            p.max(1e-300).ln()
        } else {
            (1.0 - p).max(1e-300).ln()
        };
    }
    total / (xs.len() as f64) < SEPARATED_LOSS
}

// RUNS GRADIENT DESCENT, RETURNING (ITERATIONS, STOP REASON, LEARNING RATE)
fn gradient_descent(
    params: &mut Params,
    xs: &[Vec<f64>],
    ys: &[f64],
    config: &TrainConfig,
) -> Result<(usize, StopReason, f64), ModelError> {
    const SEPARATION_CHECK_EVERY: usize = 100;

    let safe_rate = safe_learning_rate(xs);
    let learning_rate = config.learning_rate.map_or(safe_rate, |r| r.min(safe_rate));

    for i in 0..config.max_iterations {
        let (grad_w, grad_b) = gradients(params, xs, ys);
        let squared_norm: f64 = grad_w.iter().map(|g| g * g).sum::<f64>() + grad_b * grad_b;
        if !squared_norm.is_finite() {
            return Err(ModelError::NumericOverflow);
        }
        if squared_norm.sqrt() <= config.tolerance {
            return Ok((i, StopReason::Converged, learning_rate));
        }
        if i % SEPARATION_CHECK_EVERY == 0 && is_separated(params, xs, ys) {
            return Ok((i, StopReason::Separated, learning_rate));
        }

        for (w, g) in params.weights.iter_mut().zip(grad_w.iter()) {
            *w -= learning_rate * g;
        }
        params.bias -= learning_rate * grad_b;
    }

    let (grad_w, grad_b) = gradients(params, xs, ys);
    let squared_norm: f64 = grad_w.iter().map(|g| g * g).sum::<f64>() + grad_b * grad_b;
    let stop_reason = if squared_norm.sqrt() <= config.tolerance {
        StopReason::Converged
    } else if is_separated(params, xs, ys) {
        StopReason::Separated
    } else {
        StopReason::IterationLimit
    };
    Ok((config.max_iterations, stop_reason, learning_rate))
}

fn validate_data(xs: &[Vec<f64>], ys: &[f64]) -> Result<(), ModelError> {
    validate_rows(xs, ys)?;
    validate_labels(ys)?;
    if !ys.contains(&0.0) || !ys.contains(&1.0) {
        return Err(ModelError::SingleClass);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance
    }

    // CLASS 1 TENDS TO HAVE LARGER x, BUT THE MIDDLE OVERLAPS
    fn sample() -> (Vec<Vec<f64>>, Vec<f64>) {
        (
            (1..=6).map(|x| vec![x as f64]).collect(),
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        )
    }

    fn fit(xs: &[Vec<f64>], ys: &[f64]) -> (LogisticRegression, FitReport) {
        LogisticRegression::fit(xs, ys, &TrainConfig::default()).unwrap()
    }

    #[test]
    fn sigmoid_values() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!(close(sigmoid(2.0), 0.8808, 1e-4));
        assert!(close(sigmoid(-2.0), 0.1192, 1e-4));
        assert_eq!(sigmoid(1000.0), 1.0);
        assert_eq!(sigmoid(-1000.0), 0.0);
    }

    #[test]
    fn predicts_with_known_weights() {
        let model = LogisticRegression {
            weights: vec![1.0, -1.0],
            bias: 0.5,
        };
        assert!(close(
            model.predict_proba(&[2.0, 1.0]).unwrap(),
            0.8176,
            1e-4
        ));
        assert_eq!(model.predict(&[2.0, 1.0]).unwrap(), 1);
        assert_eq!(model.predict(&[0.0, 2.0]).unwrap(), 0);
    }

    #[test]
    fn starting_gradients() {
        let (xs, ys) = sample();
        let params = Params {
            weights: vec![0.0],
            bias: 0.0,
        };
        let (grad_w, grad_b) = gradients(&params, &xs, &ys);
        assert!(close(grad_w[0], -0.5833, 1e-4));
        assert!(close(grad_b, 0.0, 1e-12));
    }

    #[test]
    fn finds_the_known_solution_in_original_units() {
        let (xs, ys) = sample();
        let (model, report) = fit(&xs, &ys);
        assert_eq!(report.stop_reason, StopReason::Converged);
        assert!(close(model.weights[0], 1.214, 1e-3));
        assert!(close(model.bias, -4.249, 1e-3));
        assert_eq!(model.predict_all(&xs).unwrap(), vec![0, 0, 0, 1, 1, 1]);
        assert!(close(model.predict_proba(&[3.5]).unwrap(), 0.5, 1e-6));
    }

    #[test]
    fn stops_on_perfectly_separated_data() {
        let xs: Vec<Vec<f64>> = (1..=4).map(|x| vec![x as f64]).collect();
        let ys = vec![0.0, 0.0, 1.0, 1.0];
        let (model, report) = fit(&xs, &ys);
        assert_eq!(report.stop_reason, StopReason::Separated);
        assert!(report.train_loss < 1e-3);
        assert_eq!(model.predict_all(&xs).unwrap(), vec![0, 0, 1, 1]);
    }

    #[test]
    fn huge_learning_rate_is_capped_instead_of_diverging() {
        let (xs, ys) = sample();
        let config = TrainConfig {
            learning_rate: Some(1e6),
            ..TrainConfig::default()
        };
        let (model, report) = LogisticRegression::fit(&xs, &ys, &config).unwrap();
        assert!(report.learning_rate < 1e6);
        assert_eq!(report.stop_reason, StopReason::Converged);
        assert!(close(model.weights[0], 1.214, 1e-3));
    }

    #[test]
    fn smaller_learning_rate_is_kept() {
        let (xs, ys) = sample();
        let config = TrainConfig {
            learning_rate: Some(0.01),
            max_iterations: 10,
            ..TrainConfig::default()
        };
        let (_, report) = LogisticRegression::fit(&xs, &ys, &config).unwrap();
        assert_eq!(report.learning_rate, 0.01);
    }

    #[test]
    fn scale_and_offset_do_not_change_the_answer() {
        let (xs, ys) = sample();
        let (model, _) = fit(&xs, &ys);
        let shifted: Vec<Vec<f64>> = xs.iter().map(|x| vec![x[0] * 1e6 + 1e9]).collect();
        let (other, _) = fit(&shifted, &ys);
        assert!(close(other.weights[0] * 1e6, model.weights[0], 1e-5));
        assert!(close(
            other.predict_proba(&[3.0 * 1e6 + 1e9]).unwrap(),
            model.predict_proba(&[3.0]).unwrap(),
            1e-6
        ));
    }

    #[test]
    fn many_correlated_features_do_not_diverge() {
        let xs: Vec<Vec<f64>> = (0..60)
            .map(|i| {
                (0..20)
                    .map(|j| i as f64 + ((i * j) % 7) as f64 * 0.1)
                    .collect()
            })
            .collect();
        let ys: Vec<f64> = (0..60)
            .map(|i| ((i % 3 != 0) && i > 20) as u8 as f64)
            .collect();
        let config = TrainConfig {
            max_iterations: 2000,
            ..TrainConfig::default()
        };
        let (_, report) = LogisticRegression::fit(&xs, &ys, &config).unwrap();
        assert!(report.train_loss.is_finite());
        assert!(report.train_loss < std::f64::consts::LN_2);
    }

    #[test]
    fn duplicate_column_shares_the_weight() {
        let (xs, ys) = sample();
        let doubled: Vec<Vec<f64>> = xs.iter().map(|x| vec![x[0], x[0]]).collect();
        let (model, _) = fit(&doubled, &ys);
        assert!(close(model.weights[0], 0.607, 1e-3));
        assert!(close(model.weights[1], 0.607, 1e-3));
    }

    #[test]
    fn constant_feature_gets_zero_weight() {
        // 0.1 REPEATED HAS A TINY NONZERO STD FROM ROUNDING
        let (xs, ys) = sample();
        let with_constant: Vec<Vec<f64>> = xs.iter().map(|x| vec![0.1, x[0]]).collect();
        let (model, _) = fit(&with_constant, &ys);
        assert_eq!(model.weights[0], 0.0);
        assert!(close(model.weights[1], 1.214, 1e-3));
    }

    #[test]
    fn all_constant_features_predict_the_base_rate() {
        let xs = vec![vec![5.0]; 4];
        let ys = vec![1.0, 0.0, 0.0, 0.0];
        let (model, report) = fit(&xs, &ys);
        assert_eq!(report.stop_reason, StopReason::Converged);
        assert!(close(model.predict_proba(&[5.0]).unwrap(), 0.25, 1e-6));
    }

    #[test]
    fn more_features_than_rows_fits_training_data() {
        let xs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 1.0, 0.0]];
        let ys = vec![1.0, 0.0];
        let (model, report) = fit(&xs, &ys);
        assert_eq!(report.stop_reason, StopReason::Separated);
        assert_eq!(model.predict_all(&xs).unwrap(), vec![1, 0]);
    }

    #[test]
    fn rejects_bad_data() {
        let config = TrainConfig::default();
        let fit_err =
            |xs: &[Vec<f64>], ys: &[f64]| LogisticRegression::fit(xs, ys, &config).unwrap_err();
        assert_eq!(fit_err(&[], &[]), ModelError::EmptyData);
        assert_eq!(fit_err(&[vec![]], &[1.0]), ModelError::NoFeatures);
        assert_eq!(
            fit_err(&[vec![1.0]], &[1.0, 0.0]),
            ModelError::LengthMismatch { left: 1, right: 2 }
        );
        assert_eq!(
            fit_err(&[vec![1.0], vec![1.0, 2.0]], &[1.0, 0.0]),
            ModelError::RaggedRow {
                row: 1,
                expected: 1,
                found: 2
            }
        );
        assert_eq!(
            fit_err(&[vec![f64::NAN], vec![1.0]], &[1.0, 0.0]),
            ModelError::NonFiniteValue { row: 0 }
        );
        assert_eq!(
            fit_err(&[vec![1.0], vec![2.0]], &[0.5, 1.0]),
            ModelError::InvalidLabel { row: 0 }
        );
        assert_eq!(
            fit_err(&[vec![1.0], vec![2.0]], &[f64::NAN, 1.0]),
            ModelError::InvalidLabel { row: 0 }
        );
        assert_eq!(
            fit_err(&[vec![1.0], vec![2.0]], &[1.0, 1.0]),
            ModelError::SingleClass
        );
        assert_eq!(
            fit_err(&[vec![1e308], vec![-1e308]], &[1.0, 0.0]),
            ModelError::NumericOverflow
        );
    }

    #[test]
    fn rejects_bad_config() {
        let (xs, ys) = sample();
        for config in [
            TrainConfig {
                learning_rate: Some(0.0),
                ..TrainConfig::default()
            },
            TrainConfig {
                learning_rate: Some(f64::NAN),
                ..TrainConfig::default()
            },
            TrainConfig {
                max_iterations: 0,
                ..TrainConfig::default()
            },
            TrainConfig {
                tolerance: -1.0,
                ..TrainConfig::default()
            },
        ] {
            assert!(matches!(
                LogisticRegression::fit(&xs, &ys, &config),
                Err(ModelError::InvalidConfig(_))
            ));
        }
    }

    #[test]
    fn reports_when_stopped_by_the_limit() {
        let (xs, ys) = sample();
        let config = TrainConfig {
            max_iterations: 1,
            ..TrainConfig::default()
        };
        let (_, report) = LogisticRegression::fit(&xs, &ys, &config).unwrap();
        assert_eq!(report.stop_reason, StopReason::IterationLimit);
        assert_eq!(report.iterations, 1);
    }

    #[test]
    fn predict_rejects_bad_input() {
        let (xs, ys) = sample();
        let (model, _) = fit(&xs, &ys);
        assert!(matches!(
            model.predict(&[1.0, 2.0]),
            Err(ModelError::LengthMismatch { .. })
        ));
        assert!(matches!(
            model.predict(&[f64::NAN]),
            Err(ModelError::NonFiniteValue { .. })
        ));
        assert_eq!(
            model.predict(&[f64::MAX]).unwrap(),
            1,
            "a huge input should saturate, not fail"
        );
        let extreme = LogisticRegression {
            weights: vec![f64::MAX, f64::MAX],
            bias: 0.0,
        };
        assert_eq!(
            extreme.predict(&[f64::MAX, -f64::MAX]),
            Err(ModelError::NumericOverflow)
        );
    }
}
