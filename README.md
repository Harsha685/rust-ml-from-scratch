# Rust ML from Scratch

Basic machine learning models written from scratch in Rust, using only the standard library. I built this to learn how the algorithms work, so everything down to the CSV reader is written by hand.

| Model | Predicts | Scores |
|---|---|---|
| `LinearRegression` | A number | `mse`, `r2` |
| `LogisticRegression` | One of two classes | `log_loss`, `accuracy`, `precision`, `recall` |

Both models are trained with gradient descent, and both matched scikit-learn on the data they were checked against (see [Testing](#testing)).

## Installation

Requires Rust 1.88 or newer. Add this to your `Cargo.toml`:

```toml
[dependencies]
rust_ml_from_scratch = { git = "https://github.com/Harsha685/rust-ml-from-scratch" }
```

## Usage

```rust
use rust_ml_from_scratch::dataset::{load_csv_labeled, train_test_split};
use rust_ml_from_scratch::logistic_regression::LogisticRegression;
use rust_ml_from_scratch::metrics::accuracy;
use rust_ml_from_scratch::training::TrainConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // THE TARGET BECOMES 1.0 FOR "Gentoo" AND 0.0 FOR EVERYTHING ELSE
    let data = load_csv_labeled(
        "examples/data/penguins.csv",
        &["bill_depth_mm", "flipper_length_mm"],
        "species",
        "Gentoo",
    )?;
    let split = train_test_split(&data.xs, &data.ys)?;

    let (model, report) =
        LogisticRegression::fit(&split.train_xs, &split.train_ys, &TrainConfig::default())?;
    println!("stopped because: {:?}", report.stop_reason);

    let predictions = model.predict_all(&split.test_xs)?;
    println!("test accuracy: {}", accuracy(&predictions, &split.test_ys)?);

    // NO SCALING NEEDED: THE MODEL WORKS IN THE ORIGINAL UNITS OF YOUR DATA
    println!("class for a new penguin: {}", model.predict(&[14.0, 215.0])?);
    Ok(())
}
```

`LinearRegression` works the same way, using `load_csv` for a numeric target.

## Examples

Both examples use the [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) dataset, and both accept your own CSV file. Run one without arguments to see its default task.

```
cargo run --example penguin_mass
cargo run --example penguin_species
```

| Example | Task | Test result |
|---|---|---|
| `penguin_mass` | Predict body mass from bill and flipper measurements | R² 0.787, typical miss 352 g |
| `penguin_species` | Predict whether a penguin is a Gentoo | Accuracy 1.0 on 68 penguins |

Gentoos are easy to separate from the other species in this data, so that perfect score says more about the task than the model. The harder penguin tasks are in [Testing](#testing).

## How it works

Both models follow the same steps:

1. **Scale the features** to mean 0 and standard deviation 1, so that one learning rate suits all of them.
2. **Run gradient descent**, repeatedly moving each weight against its gradient.
3. **Convert the weights back** to the original units, so new inputs don't need scaling.

They differ in their prediction and loss:

| | Linear regression | Logistic regression |
|---|---|---|
| Prediction | `ŷ = w·x + b` | `p = sigmoid(w·x + b)` |
| Loss | Mean squared error | Log loss |
| Gradient of `wⱼ` | `(2/n) · Σ (ŷᵢ − yᵢ) · xᵢⱼ` | `(1/n) · Σ (pᵢ − yᵢ) · xᵢⱼ` |

**Learning rate:** neither model can diverge. Linear regression halves its learning rate whenever a step wouldn't lower the loss enough. Logistic regression caps its rate at a limit that's mathematically guaranteed to lower the loss.

**Stopping:** training reports one of three outcomes in `FitReport::stop_reason`:

- `Converged`: the gradient reached zero, so this is the best possible model.
- `Separated`: classification only. A line splits the training data perfectly, so the weights would otherwise grow forever. Predictions are fine, but the weights have no single correct value.
- `IterationLimit`: training ran out of iterations, so results may be inexact.

## Project layout

| File | Contents |
|---|---|
| `src/linear_regression.rs` | The linear regression model |
| `src/logistic_regression.rs` | The logistic regression model |
| `src/training.rs` | `TrainConfig`, `FitReport`, and `StopReason` |
| `src/metrics.rs` | All scoring functions |
| `src/preprocessing.rs` | Feature scaling |
| `src/dataset.rs` | The CSV reader and train/test split |
| `src/error.rs` | `ModelError` |
| `examples/` | The two penguin programs and their data |

Run `cargo doc --open` for the full API documentation.

## Testing

```
cargo test
```

The unit tests cover known solutions, malformed CSV files, invalid input, and numeric edge cases.

Both models were also compared against scikit-learn 1.9.1 as a one-off check, before the two models were combined into this crate. Those comparison scripts aren't part of this repo.

**Linear regression:**

| Check | Result |
|---|---|
| Penguin test R² | 0.78727263 for both |
| 300 random datasets with widely varying scales | All converged; predictions within 6.7 × 10⁻⁸ standard deviations of `y` |
| 1,500 corrupted CSV files | No panics |

**Logistic regression**, with scikit-learn's regularization turned off:

| Predicting | Features | Stopped because | Accuracy | scikit-learn |
|---|---|---|---|---|
| Gentoo | 3 | Separated | 1.0000 | 1.0000 |
| Adelie | 3 | Converged | 0.9853 | 0.9853 |
| Chinstrap | 3 | Converged | 0.9265 | 0.9265 |
| Male | 3 | Converged | 0.6970 | 0.6970 |
| Gentoo | 1 | Converged | 0.7059 | 0.7059 |
| Adelie | 1 | Converged | 0.9706 | 0.9706 |
| Chinstrap | 2 | Converged | 0.7647 | 0.7647 |

Of 300 random logistic datasets, 289 converged to within 1.5 × 10⁻⁴ of scikit-learn's raw outputs (`w·x + b`), and 2 stopped as separated. The other 9 hit the iteration limit and differed widely from scikit-learn. 1,500 corrupted CSV files caused no panics.

## Limitations

- **No regularization.** scikit-learn's `LogisticRegression` uses it by default, so its default weights will differ.
- **Slow on hard data.** Gradient descent can use many iterations on highly correlated features or nearly separable classes. Check `stop_reason`, since results after `IterationLimit` may be inexact. For example, at a correlation of 0.9999 between two features, linear regression weights were off by up to 7% after 100,000 iterations, while predictions stayed close.
- **More features than rows has no single answer.** Linear regression finds an exact fit, but not the same one as scikit-learn.
- **Two classes only.** Logistic regression doesn't handle more than two classes.
- **Numeric features only.** Categories such as species can only be used as a classification target.
- **The train/test split is fixed.** Every 5th row is used for testing, with no shuffling.
- **Very large values are rejected.** `fit` returns an error if a column's squared differences overflow `f64`.

## Dataset

`examples/data/penguins.csv` is the Palmer Penguins dataset, collected by Dr. Kristen Gorman at Palmer Station, Antarctica. It was released by Horst, Hill, and Gorman under the CC0 license.

## License

[MIT](LICENSE)
