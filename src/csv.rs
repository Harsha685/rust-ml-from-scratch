use std::error::Error;
use std::fmt;
use std::fs;

// EVERYTHING THAT CAN GO WRONG WHILE READING DATA
#[derive(Debug, Clone, PartialEq)]
pub enum CsvError {
    Unreadable { path: String, reason: String },
    EmptyFile,
    UnterminatedQuote,
    MissingColumn(String),
    DuplicateColumn(String),
    TargetIsFeature(String),
    NoFeatures,
    NoUsableRows { skipped: usize },
    TooFewRowsToSplit { rows: usize },
}

impl fmt::Display for CsvError {
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
            Self::TooFewRowsToSplit { rows } => {
                write!(f, "{} rows is too few to split into train and test", rows)
            }
        }
    }
}

impl Error for CsvError {}

// THE NUMBERS READ FROM A CSV
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    pub xs: Vec<Vec<f64>>,
    pub ys: Vec<f64>,
    pub skipped: usize,
}

// TRAIN AND TEST ROWS
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    pub train_xs: Vec<Vec<f64>>,
    pub train_ys: Vec<f64>,
    pub test_xs: Vec<Vec<f64>>,
    pub test_ys: Vec<f64>,
}

// READS THE NAMED COLUMNS FROM A CSV FILE
pub fn load_csv(path: &str, features: &[&str], target: &str) -> Result<Dataset, CsvError> {
    let text = fs::read_to_string(path).map_err(|e| CsvError::Unreadable {
        path: path.to_string(),
        reason: e.to_string(),
    })?;
    parse_dataset(&text, features, target)
}

// PICKS THE NAMED COLUMNS OUT OF CSV TEXT, SKIPPING ROWS THAT ARE SHORT OR HAVE
// A MISSING OR NON-NUMERIC VALUE IN THOSE COLUMNS
pub fn parse_dataset(text: &str, features: &[&str], target: &str) -> Result<Dataset, CsvError> {
    if features.is_empty() {
        return Err(CsvError::NoFeatures);
    }
    if features.contains(&target) {
        return Err(CsvError::TargetIsFeature(target.to_string()));
    }

    let records = parse_records(text)?;
    let mut records = records.iter();
    let header = records.next().ok_or(CsvError::EmptyFile)?;

    let column = |name: &str| -> Result<usize, CsvError> {
        let mut matches = header.iter().enumerate().filter(|(_, h)| h.trim() == name);
        let (index, _) = matches
            .next()
            .ok_or_else(|| CsvError::MissingColumn(name.to_string()))?;
        if matches.next().is_some() {
            return Err(CsvError::DuplicateColumn(name.to_string()));
        }
        Ok(index)
    };
    let feature_columns = features
        .iter()
        .map(|name| column(name))
        .collect::<Result<Vec<usize>, CsvError>>()?;
    let target_column = column(target)?;

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut skipped = 0;
    for record in records {
        let row: Option<Vec<f64>> = feature_columns
            .iter()
            .map(|&c| parse_number(record, c))
            .collect();
        match (row, parse_number(record, target_column)) {
            (Some(row), Some(y)) => {
                xs.push(row);
                ys.push(y);
            }
            _ => skipped += 1,
        }
    }

    if xs.is_empty() {
        return Err(CsvError::NoUsableRows { skipped });
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
fn parse_records(text: &str) -> Result<Vec<Vec<String>>, CsvError> {
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
        return Err(CsvError::UnterminatedQuote);
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

// HOLDS BACK EVERY 5TH ROW AS TEST DATA THE MODEL NEVER TRAINS ON
pub fn train_test_split(xs: &[Vec<f64>], ys: &[f64]) -> Result<Split, CsvError> {
    if xs.len() < 5 || xs.len() != ys.len() {
        return Err(CsvError::TooFewRowsToSplit { rows: xs.len() });
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
mod tests {
    use super::*;

    #[test]
    fn reads_named_columns_in_requested_order() {
        let data = parse_dataset("a,b,y\n1,2,3\n4,5,6\n", &["b", "a"], "y").unwrap();
        assert_eq!(data.xs, vec![vec![2.0, 1.0], vec![5.0, 4.0]]);
        assert_eq!(data.ys, vec![3.0, 6.0]);
        assert_eq!(data.skipped, 0);
    }

    #[test]
    fn skips_rows_with_bad_values() {
        let text = "a,y\n,1\nNA,1\nNaN,1\ninf,1\n1,-inf\n1\n2,3\n";
        let data = parse_dataset(text, &["a"], "y").unwrap();
        assert_eq!(data.xs, vec![vec![2.0]]);
        assert_eq!(data.skipped, 6);
    }

    #[test]
    fn handles_quotes() {
        let text = "name,\"a\",y\n\"Smith, John\",\"1\",2\n\"say \"\"hi\"\"\nthere\",3,4\n";
        let data = parse_dataset(text, &["a"], "y").unwrap();
        assert_eq!(data.xs, vec![vec![1.0], vec![3.0]]);
        assert_eq!(data.ys, vec![2.0, 4.0]);
    }

    #[test]
    fn handles_bom_crlf_spaces_and_blank_lines() {
        let text = "\u{feff}a , y\r\n 1 , 2 \r\n\r\n3,4";
        let data = parse_dataset(text, &["a"], "y").unwrap();
        assert_eq!(data.xs, vec![vec![1.0], vec![3.0]]);
        assert_eq!(data.ys, vec![2.0, 4.0]);
    }

    #[test]
    fn reports_bad_files() {
        assert_eq!(parse_dataset("", &["a"], "y"), Err(CsvError::EmptyFile));
        assert_eq!(parse_dataset("a,y\n", &[], "y"), Err(CsvError::NoFeatures));
        assert_eq!(
            parse_dataset("a,y\n", &["b"], "y"),
            Err(CsvError::MissingColumn("b".to_string()))
        );
        assert_eq!(
            parse_dataset("a,a,y\n1,2,3\n", &["a"], "y"),
            Err(CsvError::DuplicateColumn("a".to_string()))
        );
        assert_eq!(
            parse_dataset("a,y\n1,2\n", &["y"], "y"),
            Err(CsvError::TargetIsFeature("y".to_string()))
        );
        assert_eq!(
            parse_dataset("a,y\nx,1\n", &["a"], "y"),
            Err(CsvError::NoUsableRows { skipped: 1 })
        );
        assert_eq!(
            parse_dataset("a,y\n\"1,2\n", &["a"], "y"),
            Err(CsvError::UnterminatedQuote)
        );
        assert!(matches!(
            load_csv("/no/such/file.csv", &["a"], "y"),
            Err(CsvError::Unreadable { .. })
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
            Err(CsvError::TooFewRowsToSplit { rows: 1 })
        );
    }
}
