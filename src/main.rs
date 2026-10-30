struct LinearRegression {
    weights: Vec<f64>,
    bias: f64,
}

impl LinearRegression {
    // CONSTRUCTOR
    fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
        }
    }

    // CALCULATES ŷ
    fn predict(&self, x: &[f64]) -> f64 {
        let mut total = self.bias;

        for (u, v) in self.weights.iter().zip(x.iter()) {
            total += u * v;
        }

        total
    }

    // CALCULATES LOSS
    fn mse(predictions: &[f64], targets: &[f64]) -> f64 {
        assert_eq!(predictions.len(), targets.len());
        let mut sum = 0.0;
        for (prediction, target) in predictions.iter().zip(targets.iter()) {
            let error = prediction - target;
            sum += error * error;
        }
        sum / predictions.len() as f64
    }

    //CALCULATES GRADIENTS
    fn gradients(&self, xs: &[Vec<f64>], ys: &[f64]) -> (Vec<f64>, f64) {
        // return (gradient of w, gradient of b)
        assert_eq!(xs.len(), ys.len());
        let mut grad_w = vec![0.0; self.weights.len()];
        let mut grad_b: f64 = 0.0;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let error = self.predict(x) - y;
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

    // TAKES ONE GRADIENT DESCENT STEP
    fn step(&mut self, xs: &[Vec<f64>], ys: &[f64], learning_rate: f64) {
        let (grad_w, grad_b) = self.gradients(xs, ys);

        for (w, g) in self.weights.iter_mut().zip(grad_w.iter()) {
            *w -= learning_rate * g;
        }

        self.bias -= learning_rate * grad_b;
    }
}

fn main() {
    let xs = vec![
        vec![1.0, 2.0],
        vec![2.0, 1.0],
        vec![3.0, 3.0],
        vec![4.0, 2.0],
        vec![5.0, 4.0],
    ];
    let ys = vec![9.0, 8.0, 16.0, 15.0, 23.0];
    let mut model = LinearRegression::new(2);

    let (grad_w, grad_b) = model.gradients(&xs, &ys);
    println!("{:?} {}", grad_w, grad_b);

    for _ in 0..1000 {
        model.step(&xs, &ys, 0.01);
    }
    println!("{:?} {}", model.weights, model.bias);

    let predictions: Vec<f64> = xs.iter().map(|x| model.predict(x)).collect();
    println!("loss: {}", LinearRegression::mse(&predictions, &ys));

    println!("{}", model.predict(&[6.0, 5.0]));
}
