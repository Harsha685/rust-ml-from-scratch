//! Basic machine learning models written from scratch, using only the Rust standard library.
//!
//! Models take their data as rows of `f64` features and scale it internally, so
//! trained weights and predictions are always in the original units of your data.
//!
//! ```
//! use rust_ml_from_scratch::linear_regression::LinearRegression;
//! use rust_ml_from_scratch::training::TrainConfig;
//!
//! // y = 2x + 1
//! let xs = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
//! let ys = vec![3.0, 5.0, 7.0, 9.0];
//!
//! let (model, _report) = LinearRegression::fit(&xs, &ys, &TrainConfig::default())?;
//! let prediction = model.predict(&[10.0])?;
//! assert!((prediction - 21.0).abs() < 1e-6);
//! # Ok::<(), rust_ml_from_scratch::error::ModelError>(())
//! ```

pub mod dataset;
pub mod error;
pub mod linear_regression;
pub mod logistic_regression;
pub mod metrics;
pub mod preprocessing;
pub mod training;
