//! Feature scaling. The models scale their data internally, so you only
//! need this module if you are working with the scaled values yourself.

use crate::error::ModelError;

/// Scaled rows, plus the means and standard deviations used to scale them.
pub type Standardized = (Vec<Vec<f64>>, Vec<f64>, Vec<f64>);

// MEAN AND STD OF A COLUMN. STD IS 0 FOR A COLUMN THAT ONLY VARIES BY ROUNDING NOISE,
// SUCH AS 0.1 REPEATED, WHICH WOULD OTHERWISE BE SCALED INTO MEANINGLESS LARGE VALUES.
pub(crate) fn mean_and_std(
    values: impl Iterator<Item = f64> + Clone,
) -> Result<(f64, f64), ModelError> {
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

/// Scales each feature to mean 0 and standard deviation 1.
///
/// Returns `(scaled rows, means, standard deviations)`. Rows must all
/// have the same length, or this returns an error.
pub fn standardize(xs: &[Vec<f64>]) -> Result<Standardized, ModelError> {
    let first = xs.first().ok_or(ModelError::EmptyData)?;
    let n_features = first.len();
    for (row, x) in xs.iter().enumerate() {
        if x.len() != n_features {
            return Err(ModelError::RaggedRow {
                row,
                expected: n_features,
                found: x.len(),
            });
        }
    }

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

/// Scales one row using saved means and standard deviations.
///
/// A column with a standard deviation of 0 has one repeated value,
/// so it scales to 0 instead of dividing by 0.
///
/// # Panics
///
/// Panics if `means` or `stds` is shorter than `x`.
pub fn scale_row(x: &[f64], means: &[f64], stds: &[f64]) -> Vec<f64> {
    let mut row = Vec::with_capacity(x.len());
    for j in 0..x.len() {
        if stds[j] == 0.0 {
            row.push(0.0);
        } else {
            row.push((x[j] - means[j]) / stds[j]);
        }
    }
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance
    }

    #[test]
    fn scaling() {
        let xs = vec![
            vec![1.0, 200.0],
            vec![2.0, 100.0],
            vec![3.0, 300.0],
            vec![4.0, 200.0],
            vec![5.0, 400.0],
        ];
        let (scaled, means, stds) = standardize(&xs).unwrap();
        assert_eq!(means, vec![3.0, 240.0]);
        assert!(close(stds[1], 101.98, 0.01));
        assert!(close(scaled[0][0], -std::f64::consts::SQRT_2, 1e-12));
        assert_eq!(scale_row(&[7.0], &[1.0], &[0.0]), vec![0.0]);
    }

    #[test]
    fn standardize_rejects_bad_shapes() {
        assert_eq!(standardize(&[]), Err(ModelError::EmptyData));
        assert_eq!(
            standardize(&[vec![1.0], vec![1.0, 2.0]]),
            Err(ModelError::RaggedRow {
                row: 1,
                expected: 1,
                found: 2
            })
        );
    }
}
