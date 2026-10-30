use std::error::Error;
use std::fmt;

// EVERYTHING THAT CAN GO WRONG WITH BAD INPUT
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    EmptyData,
    NoFeatures,
    LengthMismatch {
        left: usize,
        right: usize,
    },
    RaggedRow {
        row: usize,
        expected: usize,
        found: usize,
    },
    NonFiniteValue {
        row: usize,
    },
    NumericOverflow,
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
            Self::NumericOverflow => write!(f, "values are too large to compute with"),
            Self::InvalidConfig(reason) => write!(f, "invalid config: {}", reason),
        }
    }
}

impl Error for ModelError {}

// TRAINING SETTINGS
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub learning_rate: f64, // STARTING RATE, LOWERED AUTOMATICALLY IF IT IS TOO HIGH
    pub max_iterations: usize,
    pub tolerance: f64, // STOP WHEN THE GRADIENT (ON SCALED DATA) IS THIS SMALL
    pub log_every: Option<usize>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.5,
            max_iterations: 100_000,
            tolerance: 1e-8,
            log_every: None,
        }
    }
}

// WHAT HAPPENED DURING TRAINING
#[derive(Debug, Clone)]
pub struct FitReport {
    pub iterations: usize,
    pub converged: bool,
    pub final_learning_rate: f64,
    pub train_loss: f64,
}

// A TRAINED MODEL. WEIGHTS AND BIAS ARE IN THE ORIGINAL UNITS OF THE DATA,
// SO NEW INPUTS NEVER NEED SCALING BEFORE PREDICTING.
#[derive(Debug, Clone)]
pub struct LinearRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LinearRegression {
    // SCALES THE DATA, RUNS GRADIENT DESCENT, AND CONVERTS THE RESULT BACK TO ORIGINAL UNITS
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
                converged: true,
                final_learning_rate: config.learning_rate,
                train_loss,
            };
            return Ok((model, report));
        }

        let scaled_ys: Vec<f64> = ys.iter().map(|y| (y - y_mean) / y_std).collect();
        let mut state = Params {
            weights: vec![0.0; means.len()],
            bias: 0.0,
        };
        let (iterations, converged, final_learning_rate) =
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
            converged,
            final_learning_rate,
            train_loss,
        };
        Ok((model, report))
    }

    // CALCULATES ŷ
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

    // CALCULATES ŷ FOR EVERY ROW
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

// RUNS GRADIENT DESCENT, RETURNING (ITERATIONS, CONVERGED, FINAL LEARNING RATE).
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
) -> Result<(usize, bool, f64), ModelError> {
    let mut learning_rate = config.learning_rate;

    for i in 0..config.max_iterations {
        if let Some(every) = config.log_every
            && i % every == 0
        {
            let loss = scaled_loss(&params.weights, params.bias, xs, ys);
            println!("iteration {}: scaled loss = {}", i, loss);
        }

        let (grad_w, grad_b) = gradients(params, xs, ys);
        let squared_norm: f64 = grad_w.iter().map(|g| g * g).sum::<f64>() + grad_b * grad_b;
        if !squared_norm.is_finite() {
            return Err(ModelError::NumericOverflow);
        }
        if squared_norm.sqrt() <= config.tolerance {
            return Ok((i, true, learning_rate));
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
    Ok((
        config.max_iterations,
        squared_norm.sqrt() <= config.tolerance,
        learning_rate,
    ))
}

fn validate_data(xs: &[Vec<f64>], ys: &[f64]) -> Result<(), ModelError> {
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
    for (row, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
        if x.len() != n_features {
            return Err(ModelError::RaggedRow {
                row,
                expected: n_features,
                found: x.len(),
            });
        }
        if !y.is_finite() || x.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::NonFiniteValue { row });
        }
    }
    Ok(())
}

fn validate_config(config: &TrainConfig) -> Result<(), ModelError> {
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(ModelError::InvalidConfig("learning_rate must be above 0"));
    }
    if config.max_iterations == 0 {
        return Err(ModelError::InvalidConfig("max_iterations must be above 0"));
    }
    if !config.tolerance.is_finite() || config.tolerance < 0.0 {
        return Err(ModelError::InvalidConfig("tolerance must not be negative"));
    }
    if config.log_every == Some(0) {
        return Err(ModelError::InvalidConfig("log_every must be above 0"));
    }
    Ok(())
}

// MEAN AND STD OF A COLUMN. STD IS 0 FOR A COLUMN THAT ONLY VARIES BY ROUNDING NOISE,
// SUCH AS 0.1 REPEATED, WHICH WOULD OTHERWISE BE SCALED INTO MEANINGLESS LARGE VALUES.
fn mean_and_std(values: impl Iterator<Item = f64> + Clone) -> Result<(f64, f64), ModelError> {
    let mut n = 0.0;
    let mut sum = 0.0;
    let mut largest: f64 = 0.0;
    for v in values.clone() {
        n += 1.0;
        sum += v;
        largest = largest.max(v.abs());
    }
    let mean = sum / n;

    let mut squared_diffs = 0.0;
    for v in values {
        squared_diffs += (v - mean).powi(2);
    }
    let std = (squared_diffs / n).sqrt();

    if !mean.is_finite() || !std.is_finite() {
        return Err(ModelError::NumericOverflow);
    }
    if std <= largest * 1e-12 {
        return Ok((mean, 0.0));
    }
    Ok((mean, std))
}

// SCALED ROWS, PLUS THE MEANS AND STDS USED TO SCALE THEM
type Standardized = (Vec<Vec<f64>>, Vec<f64>, Vec<f64>);

// SCALES EACH FEATURE TO MEAN 0 AND STD 1
fn standardize(xs: &[Vec<f64>]) -> Result<Standardized, ModelError> {
    let n_features = xs[0].len();
    let mut means = vec![0.0; n_features];
    let mut stds = vec![0.0; n_features];

    for j in 0..n_features {
        let (mean, std) = mean_and_std(xs.iter().map(|x| x[j]))?;
        means[j] = mean;
        stds[j] = std;
    }

    let mut scaled = Vec::with_capacity(xs.len());
    for x in xs {
        scaled.push(scale_row(x, &means, &stds));
    }

    Ok((scaled, means, stds))
}

// SCALES ONE ROW USING SAVED MEANS AND STDS
fn scale_row(x: &[f64], means: &[f64], stds: &[f64]) -> Vec<f64> {
    let mut row = Vec::with_capacity(x.len());
    for j in 0..x.len() {
        // A COLUMN WITH ONE REPEATED VALUE HAS STD 0, SO IT SCALES TO 0 INSTEAD OF DIVIDING BY 0
        if stds[j] == 0.0 {
            row.push(0.0);
        } else {
            row.push((x[j] - means[j]) / stds[j]);
        }
    }
    row
}

fn validate_scores(predictions: &[f64], targets: &[f64]) -> Result<(), ModelError> {
    if predictions.len() != targets.len() {
        return Err(ModelError::LengthMismatch {
            left: predictions.len(),
            right: targets.len(),
        });
    }
    if targets.is_empty() {
        return Err(ModelError::EmptyData);
    }
    Ok(())
}

// CALCULATES LOSS
pub fn mse(predictions: &[f64], targets: &[f64]) -> Result<f64, ModelError> {
    validate_scores(predictions, targets)?;
    let mut sum = 0.0;
    for (prediction, target) in predictions.iter().zip(targets.iter()) {
        let error = prediction - target;
        sum += error * error;
    }
    Ok(sum / predictions.len() as f64)
}

// CALCULATES R². A CONSTANT TARGET SCORES 1 IF PREDICTED PERFECTLY, OTHERWISE 0.
pub fn r2(predictions: &[f64], targets: &[f64]) -> Result<f64, ModelError> {
    validate_scores(predictions, targets)?;

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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(report.converged);
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
            learning_rate: 1e6,
            ..TrainConfig::default()
        };
        let (model, report) = LinearRegression::fit(&xs, &ys, &config).unwrap();
        assert!(report.final_learning_rate < 1e6);
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
        let model = fit(&xs, &ys);
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
                learning_rate: 0.0,
                ..TrainConfig::default()
            },
            TrainConfig {
                learning_rate: f64::NAN,
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
            TrainConfig {
                log_every: Some(0),
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
        assert!(!report.converged);
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

    #[test]
    fn scoring_edge_cases() {
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
}
