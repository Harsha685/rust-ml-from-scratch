//! Loading training data from CSV files and splitting it into train and test rows.

use std::error::Error;
use std::fmt;
use std::fs;

/// Everything that can go wrong while reading data.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DatasetError {
    Unreadable { path: String, reason: String },
    EmptyFile,
    UnterminatedQuote,
    MissingColumn(String),
    DuplicateColumn(String),
    TargetIsFeature(String),
    NoFeatures,
    NoUsableRows { skipped: usize },
    LabelNotFound(String),
    TooFewRowsToSplit { rows: usize },
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreadable { path, reason } => write!(f, "could not read {}: {}", path, reason),
            Self::EmptyFile => write!(f, "csv file is empty"),
            Self::UnterminatedQuote => write!(f, "csv file has a quote that is never closed"),
            Self::MissingColumn(name) => write!(f, "column '{}' is not in the csv header", name),
            Self::DuplicateColumn(name) => write!(f, "column '{}' appears more than once", name),
            Self::TargetIsFeature(name) => {
                write!(f, "column '{}' is used as both target and feature", name)
            }
            Self::NoFeatures => write!(f, "no feature columns were given"),
            Self::NoUsableRows { skipped } => {
                write!(f, "no usable rows ({} skipped for bad values)", skipped)
            }
            Self::LabelNotFound(label) => {
                write!(f, "no row has the label '{}' in the target column", label)
            }
            Self::TooFewRowsToSplit { rows } => {
                write!(f, "{} rows is too few to split into train and test", rows)
            }
        }
    }
}

impl Error for DatasetError {}

/// The numbers read from a CSV file.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    /// One row of features per example.
    pub xs: Vec<Vec<f64>>,
    /// The target for each example.
    pub ys: Vec<f64>,
    /// How many rows were skipped for missing or invalid values.
    pub skipped: usize,
}

/// Train and test rows.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    pub train_xs: Vec<Vec<f64>>,
    pub train_ys: Vec<f64>,
    pub test_xs: Vec<Vec<f64>>,
    pub test_ys: Vec<f64>,
}

fn read_file(path: &str) -> Result<String, DatasetError> {
    fs::read_to_string(path).map_err(|e| DatasetError::Unreadable {
        path: path.to_string(),
        reason: e.to_string(),
    })
}

/// Reads the named columns from a CSV file, for regression. The target must be numeric.
pub fn load_csv(path: &str, features: &[&str], target: &str) -> Result<Dataset, DatasetError> {
    parse_csv(&read_file(path)?, features, target)
}

/// Reads the named columns from a CSV file, for classification. The target
/// becomes 1.0 where it equals `positive`, and 0.0 otherwise.
pub fn load_csv_labeled(
    path: &str,
    features: &[&str],
    target: &str,
    positive: &str,
) -> Result<Dataset, DatasetError> {
    parse_csv_labeled(&read_file(path)?, features, target, positive)
}

/// Like [`load_csv`], but reads from CSV text instead of a file.
pub fn parse_csv(text: &str, features: &[&str], target: &str) -> Result<Dataset, DatasetError> {
    parse_columns(text, features, target, |record, column| {
        parse_number(record, column)
    })
}

/// Like [`load_csv_labeled`], but reads from CSV text instead of a file.
pub fn parse_csv_labeled(
    text: &str,
    features: &[&str],
    target: &str,
    positive: &str,
) -> Result<Dataset, DatasetError> {
    let data = parse_columns(text, features, target, |record, column| {
        record
            .get(column)
            .map(|cell| cell.trim())
            .filter(|cell| !cell.is_empty())
            .map(|cell| if cell == positive { 1.0 } else { 0.0 })
    })?;

    // A MISTYPED LABEL WOULD OTHERWISE TURN EVERY ROW INTO CLASS 0 WITHOUT ANY ERROR
    if !data.ys.contains(&1.0) {
        return Err(DatasetError::LabelNotFound(positive.to_string()));
    }
    Ok(data)
}

// PICKS THE NAMED COLUMNS OUT OF CSV TEXT, SKIPPING ROWS THAT ARE SHORT,
// HAVE A MISSING OR NON-NUMERIC FEATURE, OR HAVE NO USABLE TARGET
fn parse_columns(
    text: &str,
    features: &[&str],
    target: &str,
    read_target: impl Fn(&[String], usize) -> Option<f64>,
) -> Result<Dataset, DatasetError> {
    if features.is_empty() {
        return Err(DatasetError::NoFeatures);
    }
    if features.contains(&target) {
        return Err(DatasetError::TargetIsFeature(target.to_string()));
    }

    let records = parse_records(text)?;
    let mut records = records.iter();
    let header = records.next().ok_or(DatasetError::EmptyFile)?;

    let column = |name: &str| -> Result<usize, DatasetError> {
        let mut matches = header.iter().enumerate().filter(|(_, h)| h.trim() == name);
        let (index, _) = matches
            .next()
            .ok_or_else(|| DatasetError::MissingColumn(name.to_string()))?;
        if matches.next().is_some() {
            return Err(DatasetError::DuplicateColumn(name.to_string()));
        }
        Ok(index)
    };
    let feature_columns = features
        .iter()
        .map(|name| column(name))
        .collect::<Result<Vec<usize>, DatasetError>>()?;
    let target_column = column(target)?;

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut skipped = 0;
    for record in records {
        let row: Option<Vec<f64>> = feature_columns
            .iter()
            .map(|&c| parse_number(record, c))
            .collect();
        match (row, read_target(record, target_column)) {
            (Some(row), Some(y)) => {
                xs.push(row);
                ys.push(y);
            }
            _ => skipped += 1,
        }
    }

    if xs.is_empty() {
        return Err(DatasetError::NoUsableRows { skipped });
    }
    Ok(Dataset { xs, ys, skipped })
}

// A CELL COUNTS ONLY IF IT EXISTS AND IS A FINITE NUMBER. "NaN" AND "inf" PARSE AS f64
// BUT WOULD POISON TRAINING, SO THEY ARE TREATED AS MISSING.
fn parse_number(record: &[String], column: usize) -> Option<f64> {
    let value: f64 = record.get(column)?.trim().parse().ok()?;
    value.is_finite().then_some(value)
}

// SPLITS CSV TEXT INTO RECORDS OF CELLS. SUPPORTS QUOTED CELLS CONTAINING COMMAS, NEWLINES,
// AND ESCAPED QUOTES (""), WINDOWS LINE ENDINGS, AND A LEADING BYTE ORDER MARK.
fn parse_records(text: &str) -> Result<Vec<Vec<String>>, DatasetError> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);

    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut cell = String::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    chars.next();
                    cell.push('"');
                }
                '"' => in_quotes = false,
                _ => cell.push(c),
            }
            continue;
        }
        match c {
            '"' => in_quotes = true,
            ',' => record.push(std::mem::take(&mut cell)),
            '\r' if chars.peek() == Some(&'\n') => {}
            '\n' => {
                record.push(std::mem::take(&mut cell));
                push_record(&mut records, std::mem::take(&mut record));
            }
            _ => cell.push(c),
        }
    }
    if in_quotes {
        return Err(DatasetError::UnterminatedQuote);
    }
    record.push(cell);
    push_record(&mut records, record);

    Ok(records)
}

// BLANK LINES ARE NOT DATA
fn push_record(records: &mut Vec<Vec<String>>, record: Vec<String>) {
    if record.len() > 1 || !record[0].trim().is_empty() {
        records.push(record);
    }
}

/// Holds back every 5th row as test data the model never trains on. The rows aren't
/// shuffled, so data sorted by its target is still sampled evenly. Needs at least 5 rows.
pub fn train_test_split(xs: &[Vec<f64>], ys: &[f64]) -> Result<Split, DatasetError> {
    if xs.len() < 5 || xs.len() != ys.len() {
        return Err(DatasetError::TooFewRowsToSplit { rows: xs.len() });
    }

    let mut split = Split {
        train_xs: Vec::new(),
        train_ys: Vec::new(),
        test_xs: Vec::new(),
        test_ys: Vec::new(),
    };
    for i in 0..xs.len() {
        if i % 5 == 4 {
            split.test_xs.push(xs[i].clone());
            split.test_ys.push(ys[i]);
        } else {
            split.train_xs.push(xs[i].clone());
            split.train_ys.push(ys[i]);
        }
    }

    Ok(split)
}

#[cfg(test)]
mod numeric_tests {
    use super::*;

    #[test]
    fn reads_named_columns_in_requested_order() {
        let data = parse_csv("a,b,y\n1,2,3\n4,5,6\n", &["b", "a"], "y").unwrap();
        assert_eq!(data.xs, vec![vec![2.0, 1.0], vec![5.0, 4.0]]);
        assert_eq!(data.ys, vec![3.0, 6.0]);
        assert_eq!(data.skipped, 0);
    }

    #[test]
    fn skips_rows_with_bad_values() {
        let text = "a,y\n,1\nNA,1\nNaN,1\ninf,1\n1,-inf\n1\n2,3\n";
        let data = parse_csv(text, &["a"], "y").unwrap();
        assert_eq!(data.xs, vec![vec![2.0]]);
        assert_eq!(data.skipped, 6);
    }

    #[test]
    fn handles_quotes() {
        let text = "name,\"a\",y\n\"Smith, John\",\"1\",2\n\"say \"\"hi\"\"\nthere\",3,4\n";
        let data = parse_csv(text, &["a"], "y").unwrap();
        assert_eq!(data.xs, vec![vec![1.0], vec![3.0]]);
        assert_eq!(data.ys, vec![2.0, 4.0]);
    }

    #[test]
    fn handles_bom_crlf_spaces_and_blank_lines() {
        let text = "\u{feff}a , y\r\n 1 , 2 \r\n\r\n3,4";
        let data = parse_csv(text, &["a"], "y").unwrap();
        assert_eq!(data.xs, vec![vec![1.0], vec![3.0]]);
        assert_eq!(data.ys, vec![2.0, 4.0]);
    }

    #[test]
    fn reports_bad_files() {
        assert_eq!(parse_csv("", &["a"], "y"), Err(DatasetError::EmptyFile));
        assert_eq!(parse_csv("a,y\n", &[], "y"), Err(DatasetError::NoFeatures));
        assert_eq!(
            parse_csv("a,y\n", &["b"], "y"),
            Err(DatasetError::MissingColumn("b".to_string()))
        );
        assert_eq!(
            parse_csv("a,a,y\n1,2,3\n", &["a"], "y"),
            Err(DatasetError::DuplicateColumn("a".to_string()))
        );
        assert_eq!(
            parse_csv("a,y\n1,2\n", &["y"], "y"),
            Err(DatasetError::TargetIsFeature("y".to_string()))
        );
        assert_eq!(
            parse_csv("a,y\nx,1\n", &["a"], "y"),
            Err(DatasetError::NoUsableRows { skipped: 1 })
        );
        assert_eq!(
            parse_csv("a,y\n\"1,2\n", &["a"], "y"),
            Err(DatasetError::UnterminatedQuote)
        );
        assert!(matches!(
            load_csv("/no/such/file.csv", &["a"], "y"),
            Err(DatasetError::Unreadable { .. })
        ));
    }

    #[test]
    fn split_keeps_every_row_once() {
        let xs: Vec<Vec<f64>> = (0..12).map(|i| vec![i as f64]).collect();
        let ys: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let split = train_test_split(&xs, &ys).unwrap();
        assert_eq!(split.test_ys, vec![4.0, 9.0]);
        assert_eq!(split.train_ys.len(), 10);
        assert_eq!(split.train_xs.len(), 10);
        assert_eq!(split.test_xs.len(), 2);
    }

    #[test]
    fn split_rejects_too_little_data() {
        assert_eq!(
            train_test_split(&[vec![1.0]], &[1.0]),
            Err(DatasetError::TooFewRowsToSplit { rows: 1 })
        );
    }
}

#[cfg(test)]
mod labeled_tests {
    use super::*;

    #[test]
    fn reads_named_columns_in_requested_order() {
        let data = parse_csv_labeled("a,b,y\n1,2,yes\n4,5,no\n", &["b", "a"], "y", "yes").unwrap();
        assert_eq!(data.xs, vec![vec![2.0, 1.0], vec![5.0, 4.0]]);
        assert_eq!(data.ys, vec![1.0, 0.0]);
        assert_eq!(data.skipped, 0);
    }

    #[test]
    fn turns_labels_into_classes() {
        let text = "a,species\n1,Gentoo\n2,Adelie\n3,Chinstrap\n4, Gentoo \n";
        let data = parse_csv_labeled(text, &["a"], "species", "Gentoo").unwrap();
        assert_eq!(data.ys, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn numeric_labels_work() {
        let data = parse_csv_labeled("a,y\n1,1\n2,0\n", &["a"], "y", "1").unwrap();
        assert_eq!(data.ys, vec![1.0, 0.0]);
    }

    #[test]
    fn skips_rows_with_bad_values() {
        let text = "a,y\n,yes\nNA,yes\nNaN,yes\ninf,yes\n1,\n1,  \n1\n2,yes\n";
        let data = parse_csv_labeled(text, &["a"], "y", "yes").unwrap();
        assert_eq!(data.xs, vec![vec![2.0]]);
        assert_eq!(data.skipped, 7);
    }

    #[test]
    fn handles_quotes() {
        let text = "name,\"a\",y\n\"Smith, John\",\"1\",yes\n\"say \"\"hi\"\"\nthere\",3,no\n";
        let data = parse_csv_labeled(text, &["a"], "y", "yes").unwrap();
        assert_eq!(data.xs, vec![vec![1.0], vec![3.0]]);
        assert_eq!(data.ys, vec![1.0, 0.0]);
    }

    #[test]
    fn handles_bom_crlf_spaces_and_blank_lines() {
        let text = "\u{feff}a , y\r\n 1 , yes \r\n\r\n3,no";
        let data = parse_csv_labeled(text, &["a"], "y", "yes").unwrap();
        assert_eq!(data.xs, vec![vec![1.0], vec![3.0]]);
        assert_eq!(data.ys, vec![1.0, 0.0]);
    }

    #[test]
    fn reports_bad_files() {
        let parse = |text, features, target| parse_csv_labeled(text, features, target, "yes");
        assert_eq!(parse("", &["a"], "y"), Err(DatasetError::EmptyFile));
        assert_eq!(parse("a,y\n", &[], "y"), Err(DatasetError::NoFeatures));
        assert_eq!(
            parse("a,y\n", &["b"], "y"),
            Err(DatasetError::MissingColumn("b".to_string()))
        );
        assert_eq!(
            parse("a,a,y\n1,2,yes\n", &["a"], "y"),
            Err(DatasetError::DuplicateColumn("a".to_string()))
        );
        assert_eq!(
            parse("a,y\n1,yes\n", &["y"], "y"),
            Err(DatasetError::TargetIsFeature("y".to_string()))
        );
        assert_eq!(
            parse("a,y\nx,yes\n", &["a"], "y"),
            Err(DatasetError::NoUsableRows { skipped: 1 })
        );
        assert_eq!(
            parse("a,y\n\"1,yes\n", &["a"], "y"),
            Err(DatasetError::UnterminatedQuote)
        );
        assert!(matches!(
            load_csv_labeled("/no/such/file.csv", &["a"], "y", "yes"),
            Err(DatasetError::Unreadable { .. })
        ));
    }

    #[test]
    fn reports_a_label_that_never_appears() {
        assert_eq!(
            parse_csv_labeled("a,s\n1,Gentoo\n", &["a"], "s", "gentoo"),
            Err(DatasetError::LabelNotFound("gentoo".to_string()))
        );
    }
}
