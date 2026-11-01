//! Predicts whether a penguin is a Gentoo from its bill and flipper measurements.
//!
//! Run it with `cargo run --example penguin_species`, or pass your own data: `cargo run
//! --example penguin_species -- <csv file> <target column> <positive label> <feature column>...`

use rust_ml_from_scratch::dataset::{load_csv_labeled, train_test_split};
use rust_ml_from_scratch::logistic_regression::LogisticRegression;
use rust_ml_from_scratch::metrics::{accuracy, log_loss, precision, recall};
use rust_ml_from_scratch::training::{StopReason, TrainConfig};
use std::env;
use std::error::Error;
use std::process::ExitCode;

const DEFAULT_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data/penguins.csv");
const DEFAULT_TARGET: &str = "species";
const DEFAULT_POSITIVE: &str = "Gentoo";
const DEFAULT_FEATURES: [&str; 3] = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"];

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {}", error);
            eprintln!(
                "usage: cargo run --example penguin_species -- [<csv file> <target column> <positive label> <feature column>...]"
            );
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    // DEFAULTS TO PREDICTING WHETHER A PENGUIN IS A GENTOO FROM ITS BILL AND FLIPPER MEASUREMENTS
    let args: Vec<String> = env::args().skip(1).collect();
    let (path, target, positive, features) = match args.as_slice() {
        [] => (
            DEFAULT_CSV,
            DEFAULT_TARGET,
            DEFAULT_POSITIVE,
            DEFAULT_FEATURES.to_vec(),
        ),
        [path, target, positive, features @ ..] if !features.is_empty() => (
            path.as_str(),
            target.as_str(),
            positive.as_str(),
            features.iter().map(String::as_str).collect(),
        ),
        _ => return Err("expected a csv file, target, positive label, and features".into()),
    };

    let data = load_csv_labeled(path, &features, target, positive)?;
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
        LogisticRegression::fit(&split.train_xs, &split.train_ys, &TrainConfig::default())?;
    println!(
        "stopped after {} iterations (learning rate: {}, train loss: {})",
        report.iterations, report.learning_rate, report.train_loss
    );
    match report.stop_reason {
        StopReason::Converged => println!("training converged"),
        StopReason::Separated => {
            println!("the classes are perfectly separable, so the weights have no final value")
        }
        _ => eprintln!("warning: training hit the iteration limit, so results may be inexact"),
    }
    println!("features = {:?}", features);
    println!("weights = {:?}, bias = {:?}", model.weights, model.bias);

    let probabilities = model.predict_proba_all(&split.test_xs)?;
    let predictions = model.predict_all(&split.test_xs)?;
    println!(
        "test log loss: {}",
        log_loss(&probabilities, &split.test_ys)?
    );
    println!("test accuracy: {}", accuracy(&predictions, &split.test_ys)?);
    println!(
        "test precision: {}",
        precision(&predictions, &split.test_ys)?
    );
    println!("test recall: {}", recall(&predictions, &split.test_ys)?);

    Ok(())
}
