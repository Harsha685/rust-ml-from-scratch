//! Linear regression trained with gradient descent, for predicting a number.

use crate::error::ModelError;
use crate::metrics::mse;
use crate::preprocessing::{mean_and_std, standardize};
use crate::training::{FitReport, StopReason, TrainConfig, validate_config, validate_rows};

// WHERE THE LEARNING RATE STARTS WHEN THE CONFIG DOESN'T SET ONE
const DEFAULT_LEARNING_RATE: f64 = 0.5;

/// A trained linear regression model. Weights and bias are in the original
/// units of the data, so new inputs never need scaling before predicting.
#[derive(Debug, Clone)]
pub struct LinearRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LinearRegression {
    /// Trains a model by scaling the data, running gradient descent, and
    /// converting the result back to original units.
    ///
    /// `xs` holds one row of features per example, and `ys` holds their target values.
    /// Returns an error for empty or malformed data, `NaN` or infinite values, or invalid settings.
    pub fn fit(
        xs: &[Vec<f64>],
        ys: &[f64],
        config: &TrainConfig,
    ) -> Result<(Self, FitReport), ModelError> {
        validate_data(xs, ys)?;
        validate_config(config)?;

        let (scaled_xs, means, stds) = standardize(xs)?;
        let (y_mean, y_std) = mean_and_std(ys.iter().copied())?;

        // A CONSTANT TARGET IS FIT EXACTLY BY ITS MEAN
        if y_std == 0.0 {
            let model = Self {
                weights: vec![0.0; means.len()],
                bias: y_mean,
            };
            let train_loss = mse(&model.predict_all(xs)?, ys)?;
            let report = FitReport {
                iterations: 0,
                stop_reason: StopReason::Converged,
                learning_rate: config.learning_rate.unwrap_or(DEFAULT_LEARNING_RATE),
                train_loss,
            };
            return Ok((model, report));
        }

        let scaled_ys: Vec<f64> = ys.iter().map(|y| (y - y_mean) / y_std).collect();
        let mut state = Params {
            weights: vec![0.0; means.len()],
            bias: 0.0,
        };
        let (iterations, stop_reason, learning_rate) =
            gradient_descent(&mut state, &scaled_xs, &scaled_ys, config)?;

        // UNDO THE SCALING: ŷ = y_mean + y_std · (b + Σ w · (x − mean) / std)
        let mut weights = vec![0.0; means.len()];
        let mut bias = y_mean + y_std * state.bias;
        for j in 0..weights.len() {
            if stds[j] != 0.0 {
                weights[j] = y_std * state.weights[j] / stds[j];
                bias -= weights[j] * means[j];
            }
        }
        if !bias.is_finite() || weights.iter().any(|w| !w.is_finite()) {
            return Err(ModelError::NumericOverflow);
        }

        let model = Self { weights, bias };
        let train_loss = mse(&model.predict_all(xs)?, ys)?;
        let report = FitReport {
            iterations,
            stop_reason,
            learning_rate,
            train_loss,
        };
        Ok((model, report))
    }

    /// Predicts the target for one row of features.
    pub fn predict(&self, x: &[f64]) -> Result<f64, ModelError> {
        if x.len() != self.weights.len() {
            return Err(ModelError::LengthMismatch {
                left: x.len(),
                right: self.weights.len(),
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::NonFiniteValue { row: 0 });
        }
        Ok(evaluate(&self.weights, self.bias, x))
    }

    /// Predicts the target for every row.
    pub fn predict_all(&self, xs: &[Vec<f64>]) -> Result<Vec<f64>, ModelError> {
        xs.iter().map(|x| self.predict(x)).collect()
    }
}

// WEIGHTS AND BIAS WHILE TRAINING, IN SCALED UNITS
struct Params {
    weights: Vec<f64>,
    bias: f64,
}

// w·x + b
fn evaluate(weights: &[f64], bias: f64, x: &[f64]) -> f64 {
    let mut total = bias;
    for (u, v) in weights.iter().zip(x.iter()) {
        total += u * v;
    }
    total
}

fn scaled_loss(weights: &[f64], bias: f64, xs: &[Vec<f64>], ys: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let error = evaluate(weights, bias, x) - y;
        sum += error * error;
    }
    sum / xs.len() as f64
}

//CALCULATES GRADIENTS
fn gradients(params: &Params, xs: &[Vec<f64>], ys: &[f64]) -> (Vec<f64>, f64) {
    // return (gradient of w, gradient of b)
    let mut grad_w = vec![0.0; params.weights.len()];
    let mut grad_b: f64 = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let error = evaluate(&params.weights, params.bias, x) - y;
        for j in 0..x.len() {
            grad_w[j] += error * x[j];
        }
        grad_b += error;
    }
    let n = xs.len() as f64;
    for g in grad_w.iter_mut() {
        *g = 2.0 * *g / n;
    }
    (grad_w, 2.0 * grad_b / n)
}

// RUNS GRADIENT DESCENT, RETURNING (ITERATIONS, STOP REASON, FINAL LEARNING RATE).
// A STEP IS ONLY TAKEN IF IT LOWERS THE LOSS ENOUGH (THE ARMIJO CONDITION). OTHERWISE THE
// LEARNING RATE IS HALVED FIRST, SO THE LOSS CAN NEVER DIVERGE.
//
// THE LOSS IS QUADRATIC, SO A STEP OF SIZE lr CHANGES IT BY EXACTLY
//     −lr · |g|² + lr² · mean(d²)
// WHERE g IS THE GRADIENT AND d IS HOW MUCH EACH PREDICTION MOVES PER UNIT OF STEP.
// THE CONDITION "DROP BY AT LEAST lr · |g|² / 2" THEN BECOMES lr · mean(d²) ≤ |g|² / 2.
// CHECKING THIS DIRECTLY, RATHER THAN COMPARING THE LOSS BEFORE AND AFTER, STILL WORKS
// NEAR THE MINIMUM WHERE THE TWO LOSSES ARE TOO CLOSE FOR f64 TO TELL APART.
fn gradient_descent(
    params: &mut Params,
    xs: &[Vec<f64>],
    ys: &[f64],
    config: &TrainConfig,
) -> Result<(usize, StopReason, f64), ModelError> {
    let mut learning_rate = config.learning_rate.unwrap_or(DEFAULT_LEARNING_RATE);

    for i in 0..config.max_iterations {
        let (grad_w, grad_b) = gradients(params, xs, ys);
        let squared_norm: f64 = grad_w.iter().map(|g| g * g).sum::<f64>() + grad_b * grad_b;
        if !squared_norm.is_finite() {
            return Err(ModelError::NumericOverflow);
        }
        if squared_norm.sqrt() <= config.tolerance {
            return Ok((i, StopReason::Converged, learning_rate));
        }

        let mean_squared_move = scaled_loss(&grad_w, grad_b, xs, &vec![0.0; ys.len()]);
        if !mean_squared_move.is_finite() {
            return Err(ModelError::NumericOverflow);
        }
        while learning_rate * mean_squared_move > 0.5 * squared_norm {
            learning_rate *= 0.5;
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
    } else {
        StopReason::IterationLimit
    };
    Ok((config.max_iterations, stop_reason, learning_rate))
}

fn validate_data(xs: &[Vec<f64>], ys: &[f64]) -> Result<(), ModelError> {
    validate_rows(xs, ys)?;
    if let Some(row) = ys.iter().position(|y| !y.is_finite()) {
        return Err(ModelError::NonFiniteValue { row });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::r2;

    fn fit(xs: &[Vec<f64>], ys: &[f64]) -> LinearRegression {
        LinearRegression::fit(xs, ys, &TrainConfig::default())
            .unwrap()
            .0
    }

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance * (1.0 + b.abs())
    }

    // y = 2·x₁ + 0.03·x₂ + 1
    fn sample() -> (Vec<Vec<f64>>, Vec<f64>) {
        (
            vec![
                vec![1.0, 200.0],
                vec![2.0, 100.0],
                vec![3.0, 300.0],
                vec![4.0, 200.0],
                vec![5.0, 400.0],
            ],
            vec![9.0, 8.0, 16.0, 15.0, 23.0],
        )
    }

    #[test]
    fn recovers_known_line_in_original_units() {
        let (xs, ys) = sample();
        let (model, report) = LinearRegression::fit(&xs, &ys, &TrainConfig::default()).unwrap();
        assert_eq!(report.stop_reason, StopReason::Converged);
        assert!(close(model.weights[0], 2.0, 1e-6));
        assert!(close(model.weights[1], 0.03, 1e-6));
        assert!(close(model.bias, 1.0, 1e-6));
        assert!(close(model.predict(&[6.0, 500.0]).unwrap(), 28.0, 1e-6));
    }

    #[test]
    fn matches_exact_answer_with_noise() {
        // EXACT LEAST-SQUARES SOLUTION FOR THIS DATA, ALSO CONFIRMED WITH scikit-learn
        let xs = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let ys = vec![2.2, 4.1, 6.3, 7.8, 10.1];
        let model = fit(&xs, &ys);
        assert!(close(model.weights[0], 1.95, 1e-6));
        assert!(close(model.bias, 0.25, 1e-6));
    }

    #[test]
    fn huge_learning_rate_recovers_instead_of_diverging() {
        let (xs, ys) = sample();
        let config = TrainConfig {
            learning_rate: Some(1e6),
            ..TrainConfig::default()
        };
        let (model, report) = LinearRegression::fit(&xs, &ys, &config).unwrap();
        assert!(report.learning_rate < 1e6);
        assert!(close(model.weights[0], 2.0, 1e-6));
    }

    #[test]
    fn many_correlated_features_do_not_diverge() {
        let xs: Vec<Vec<f64>> = (0..50)
            .map(|i| {
                (0..20)
                    .map(|j| i as f64 + ((i * j) % 7) as f64 * 0.1)
                    .collect()
            })
            .collect();
        let ys: Vec<f64> = (0..50).map(|i| 2.0 * i as f64).collect();
        let config = TrainConfig {
            max_iterations: 100_000,
            ..TrainConfig::default()
        };
        let (model, _) = LinearRegression::fit(&xs, &ys, &config).unwrap();
        assert!(r2(&model.predict_all(&xs).unwrap(), &ys).unwrap() > 0.999);
    }

    #[test]
    fn large_target_offset_is_accurate() {
        let xs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let ys: Vec<f64> = (0..20).map(|i| 1e6 + 3.0 * i as f64).collect();
        let model = fit(&xs, &ys);
        assert!(close(model.weights[0], 3.0, 1e-6));
        assert!(close(model.bias, 1e6, 1e-6));
    }

    #[test]
    fn duplicate_column_still_predicts_correctly() {
        let xs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, i as f64]).collect();
        let ys: Vec<f64> = (0..10).map(|i| 4.0 * i as f64 + 1.0).collect();
        let model = fit(&xs, &ys);
        assert!(close(model.weights[0], 2.0, 1e-6));
        assert!(close(model.predict(&[20.0, 20.0]).unwrap(), 81.0, 1e-6));
    }

    #[test]
    fn constant_feature_gets_zero_weight() {
        // 0.1 REPEATED HAS A TINY NONZERO STD FROM ROUNDING
        let xs: Vec<Vec<f64>> = (0..7).map(|i| vec![0.1, i as f64]).collect();
        let ys: Vec<f64> = (0..7).map(|i| 2.0 * i as f64 + 5.0).collect();
        let model = fit(&xs, &ys);
        assert_eq!(model.weights[0], 0.0);
        assert!(close(model.weights[1], 2.0, 1e-6));
        assert!(close(model.bias, 5.0, 1e-6));
    }

    #[test]
    fn constant_target_predicts_its_value() {
        let xs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let model = fit(&xs, &[7.0, 7.0, 7.0]);
        assert_eq!(model.weights[0], 0.0);
        assert_eq!(model.bias, 7.0);
    }

    #[test]
    fn single_row_is_handled() {
        let model = fit(&[vec![3.0]], &[9.0]);
        assert_eq!(model.predict(&[3.0]).unwrap(), 9.0);
    }

    #[test]
    fn more_features_than_rows_fits_training_data() {
        let xs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 1.0, 0.0]];
        let ys = vec![1.0, 2.0];
        let model = fit(&xs, &ys);
        assert!(r2(&model.predict_all(&xs).unwrap(), &ys).unwrap() > 0.999999);
    }

    #[test]
    fn rejects_bad_data() {
        let config = TrainConfig::default();
        let fit_err =
            |xs: &[Vec<f64>], ys: &[f64]| LinearRegression::fit(xs, ys, &config).unwrap_err();
        assert_eq!(fit_err(&[], &[]), ModelError::EmptyData);
        assert_eq!(fit_err(&[vec![]], &[1.0]), ModelError::NoFeatures);
        assert_eq!(
            fit_err(&[vec![1.0]], &[1.0, 2.0]),
            ModelError::LengthMismatch { left: 1, right: 2 }
        );
        assert_eq!(
            fit_err(&[vec![1.0], vec![1.0, 2.0]], &[1.0, 2.0]),
            ModelError::RaggedRow {
                row: 1,
                expected: 1,
                found: 2
            }
        );
        assert_eq!(
            fit_err(&[vec![f64::NAN]], &[1.0]),
            ModelError::NonFiniteValue { row: 0 }
        );
        assert_eq!(
            fit_err(&[vec![1.0]], &[f64::INFINITY]),
            ModelError::NonFiniteValue { row: 0 }
        );
        assert_eq!(
            fit_err(&[vec![1e308], vec![-1e308]], &[1.0, 2.0]),
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
                LinearRegression::fit(&xs, &ys, &config),
                Err(ModelError::InvalidConfig(_))
            ));
        }
    }

    #[test]
    fn reports_when_not_converged() {
        let (xs, ys) = sample();
        let config = TrainConfig {
            max_iterations: 1,
            ..TrainConfig::default()
        };
        let (_, report) = LinearRegression::fit(&xs, &ys, &config).unwrap();
        assert_eq!(report.stop_reason, StopReason::IterationLimit);
        assert_eq!(report.iterations, 1);
    }

    #[test]
    fn predict_rejects_bad_input() {
        let (xs, ys) = sample();
        let model = fit(&xs, &ys);
        assert!(matches!(
            model.predict(&[1.0]),
            Err(ModelError::LengthMismatch { .. })
        ));
        assert!(matches!(
            model.predict(&[1.0, f64::NAN]),
            Err(ModelError::NonFiniteValue { .. })
        ));
    }
}
