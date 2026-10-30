# Linear Regression from Scratch in Rust

A multiple linear regression model trained with gradient descent, using only the Rust standard library. It includes its own CSV reader, feature scaling, and scoring, and it matches scikit-learn's results on well-conditioned data.

## Quick start

Requires Rust 1.88 or newer (the project uses the 2024 edition and let chains).

```
cargo run
```

This trains on the included penguin dataset, predicting body mass from bill and flipper measurements:

```
loaded 342 rows, skipped 2 with missing values
274 training rows, 68 test rows
stopped after 139 iterations (converged: true, learning rate: 0.25, train loss: 160702.70046784967)
features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]
weights = [2.5638399615625396, 32.8774788708226, 52.01954455589183], bias = -6924.6946614284625
train R²: 0.7549520057958007
test R²: 0.7872726327187334
typical test miss: 351.9291333451723
```

To use your own data, pass a CSV file, the column to predict, and one or more feature columns:

```
cargo run -- data/penguins.csv body_mass_g flipper_length_mm
```

## How it works

**Model:** the prediction is a weighted sum of the features plus a bias.

```
ŷ = w₁·x₁ + w₂·x₂ + ... + b
```

**Loss:** mean squared error measures how wrong the predictions are.

```
MSE = (1/n) · Σ (ŷᵢ − yᵢ)²
```

**Training:** gradient descent repeatedly moves each value against its gradient.

```
gradient of wⱼ = (2/n) · Σ (ŷᵢ − yᵢ) · xᵢⱼ
gradient of b  = (2/n) · Σ (ŷᵢ − yᵢ)

wⱼ = wⱼ − learning_rate · gradient of wⱼ
b  = b  − learning_rate · gradient of b
```

Three things make this reliable on real data:

- **Scaling:** features and the target are scaled to mean 0 and standard deviation 1 before training. The final weights are converted back to original units, so you never need to scale inputs yourself.
- **Automatic learning rate:** before each step, the model checks that the step lowers the loss enough, and halves the learning rate if it wouldn't. Training can't diverge, even from a learning rate that is far too large.
- **Stopping rule:** training stops when the gradient is close to 0, and reports whether that happened before the iteration limit.

## Using it in code

```rust
let data = load_csv("data/penguins.csv", &["flipper_length_mm"], "body_mass_g")?;
let split = train_test_split(&data.xs, &data.ys)?;

let (model, report) =
    LinearRegression::fit(&split.train_xs, &split.train_ys, &TrainConfig::default())?;

let predictions = model.predict_all(&split.test_xs)?;
println!("test R²: {}", r2(&predictions, &split.test_ys)?);
println!("converged: {}", report.converged);
```

`TrainConfig` sets the starting learning rate, the iteration limit, the stopping tolerance, and optional loss logging.

## Project layout

| File | Contents |
|---|---|
| `src/linear_regression.rs` | The model, training, scaling, `mse`, `r2`, and their tests |
| `src/csv.rs` | The CSV reader, train/test split, and their tests |
| `src/main.rs` | The command-line program |
| `data/penguins.csv` | The example dataset |

## Testing

```
cargo test
```

The 22 unit tests cover known solutions, malformed CSV files, invalid input, and numeric edge cases.

The model was also compared against scikit-learn 1.9.1 as a one-off check. That comparison script isn't part of this repo.

| Check | Result |
|---|---|
| Penguin test R² | 0.78727263 for both |
| 300 random datasets with widely varying scales | All converged; predictions differed by at most 6.7 × 10⁻⁸ standard deviations of `y` |
| 1,500 corrupted CSV files | No panics |

## Behavior on difficult data

| Data | What happens |
|---|---|
| Missing or non-numeric values, `NaN`, `inf`, short rows | The row is skipped and counted |
| Quoted cells, Windows line endings, byte order marks | Handled |
| `NaN` or infinity passed directly to `fit` or `predict` | Returns an error |
| A constant feature | Gets a weight of 0 |
| A constant target | Predicts that constant |
| Duplicate features | Predictions are correct; the weight is shared between the copies |
| Features on extremely different scales (10⁻⁶ to 10⁶) | Works. On this data, scikit-learn's raw fit dropped a feature, scoring R² 0.16 to this model's 0.92, unless the data was scaled first |

## Limitations

- **Very highly correlated features train slowly.** At a correlation of 0.9999 between two features, training hit the 100,000-iteration limit with `converged: false`, and the weights differed from the exact solution by up to 7%. Test R² was still within 0.0003 of scikit-learn's. Exact solvers don't have this problem.
- **More features than rows has no single answer.** Many exact fits exist, and this model finds a different one from scikit-learn, so test predictions differ.
- **Precision is lower than an exact solver.** With the default tolerance, predictions matched scikit-learn to within about 10⁻⁷ standard deviations of `y`, and weights to about 5 to 8 significant digits.
- **Only numeric columns are used.** Categories such as species aren't encoded.
- **Outliers aren't handled.** Squared error lets one extreme value pull the whole line.
- **Very large values are rejected.** If a column's squared differences overflow `f64`, for example at around 10¹⁶⁰ and above, `fit` returns an error.
- **The train/test split is fixed.** Every 5th row is used for testing, with no shuffling.

## Dataset

`data/penguins.csv` is the Palmer Penguins dataset, collected by Dr. Kristen Gorman at Palmer Station, Antarctica. It was released by Horst, Hill, and Gorman under the CC0 license in the [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/) package.
