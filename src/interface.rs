use ndarray::Array1;
use ordered_float::NotNan;

pub trait Problem {
    fn evaluate(&self, x: &Array1<f64>) -> NotNan<f64>;
}
