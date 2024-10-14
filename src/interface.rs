use anyhow::Result as AnyResult;
use ndarray::Array1;
use ordered_float::NotNan;

pub trait Problem {
    fn variable_dimension(&self) -> usize;
    fn evaluate(&self, x: &Array1<f64>) -> AnyResult<NotNan<f64>>;
}

pub trait Optimizer {
    type State;

    fn initialize(&self, problem: &dyn Problem) -> AnyResult<Self::State>;

    fn step(
        &self,
        problem: &dyn Problem,
        state: Self::State,
    ) -> AnyResult<(NotNan<f64>, Array1<f64>, Self::State)>;

    fn optimize(
        &self,
        problem: &dyn Problem,
        state: Option<Self::State>,
    ) -> AnyResult<(NotNan<f64>, Array1<f64>, Self::State)> {
        let state = if let Some(state) = state {
            state
        } else {
            self.initialize(problem)?
        };
        let (value, solution, state) = self.step(problem, state)?;
        Ok((value, solution, state))
    }
}
