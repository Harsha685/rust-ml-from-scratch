mod csv;
mod linear_regression;

use csv::{load_csv, train_test_split};
use linear_regression::{LinearRegression, TrainConfig, mse, r2};
use std::env;
use std::error::Error;
use std::process::ExitCode;

const DEFAULT_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/penguins.csv");
const DEFAULT_TARGET: &str = "body_mass_g";
const DEFAULT_FEATURES: [&str; 3] = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"];

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {}", error);
            eprintln!("usage: cargo run -- [<csv file> <target column> <feature column>...]");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    // DEFAULTS TO PREDICTING A PENGUIN'S BODY MASS FROM ITS BILL AND FLIPPER MEASUREMENTS
    let args: Vec<String> = env::args().skip(1).collect();
    let (path, target, features) = match args.as_slice() {
        [] => (DEFAULT_CSV, DEFAULT_TARGET, DEFAULT_FEATURES.to_vec()),
        [path, target, features @ ..] if !features.is_empty() => (
            path.as_str(),
            target.as_str(),
            features.iter().map(String::as_str).collect(),
        ),
        _ => return Err("expected a csv file, a target column, and at least one feature".into()),
    };

    let data = load_csv(path, &features, target)?;
    println!(
        "loaded {} rows, skipped {} with missing values",
        data.ys.len(),
        data.skipped
    );

    let split = train_test_split(&data.xs, &data.ys)?;
    println!(
        "{} training rows, {} test rows",
        split.train_ys.len(),
        split.test_ys.len()
    );

    let (model, report) =
        LinearRegression::fit(&split.train_xs, &split.train_ys, &TrainConfig::default())?;
    println!(
        "stopped after {} iterations (converged: {}, learning rate: {}, train loss: {})",
        report.iterations, report.converged, report.final_learning_rate, report.train_loss
    );
    if !report.converged {
        eprintln!("warning: training hit the iteration limit, so results may be inexact");
    }
    println!("features = {:?}", features);
    println!("weights = {:?}, bias = {:?}", model.weights, model.bias);

    let train_predictions = model.predict_all(&split.train_xs)?;
    let test_predictions = model.predict_all(&split.test_xs)?;
    println!("train R²: {}", r2(&train_predictions, &split.train_ys)?);
    println!("test R²: {}", r2(&test_predictions, &split.test_ys)?);
    println!(
        "typical test miss: {}",
        mse(&test_predictions, &split.test_ys)?.sqrt()
    );

    Ok(())
}
