struct LinearRegression {
    weights: f64,
    bias: f64,
}

impl LinearRegression {

    // CONSTRUCTOR
    fn new() -> Self {
        Self {
            weights: 0.0,
            bias: 0.0,
        }
    }

    // CALCULATES ŷ
    fn predict(&self, x: f64) -> f64 {
        self.weights * x + self.bias
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

    // CALCULATES GRADIENTS
    fn gradients(&self, xs: &[f64], ys: &[f64]) -> (f64, f64) {
        // return (gradient of w, gradient of b)
        assert_eq!(xs.len(), ys.len());
        let mut grad_w: f64 =0.0;
        let mut grad_b: f64 =0.0; 
        for (x, y) in xs.iter().zip(ys.iter()) {
            let error = self.predict(*x) - y;
            grad_w += error * x;
            grad_b += error;
        }
        let n = xs.len() as f64;
        (2.0 * grad_w / n, 2.0 * grad_b / n)
    }

    fn step(&mut self, xs: &[f64], ys: &[f64], learning_rate: f64) {
        let (grad_w, grad_b) = self.gradients(xs, ys);

        self.weights -= learning_rate * grad_w;
        self.bias -= learning_rate * grad_b;
    }
}

fn main() {
    let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let ys = vec![3.0, 5.0, 7.0, 9.0, 11.0];
    let mut model = LinearRegression::new();

    for i in 0..1000 {
        if i % 100 == 0 {
            let predictions: Vec<f64> = xs.iter().map(|&x| model.predict(x)).collect();
            println!("iteration {}: loss = {}", i, LinearRegression::mse(&predictions, &ys));
        }
        model.step(&xs, &ys, 0.0832);
    }

    println!("weights = {}, bias = {}", model.weights, model.bias);
    println!("prediction for x = 6: {}", model.predict(6.0));
}