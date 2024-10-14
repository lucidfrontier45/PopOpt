use anyhow::Result as AnyResult;
use ndarray::Array1;
use ordered_float::NotNan;

pub type Variable = Array1<f64>;
pub type Score = NotNan<f64>;

pub trait Problem {
    fn variable_dimension(&self) -> usize;
    fn evaluate(&self, x: &Variable) -> AnyResult<Score>;
}

pub trait Optimizer {
    type State;

    fn initialize(&self, problem: &dyn Problem) -> AnyResult<Self::State>;

    fn step(
        &self,
        problem: &dyn Problem,
        state: Self::State,
    ) -> AnyResult<(Score, Variable, Self::State)>;

    fn optimize(
        &self,
        problem: &dyn Problem,
        state: Option<Self::State>,
    ) -> AnyResult<(Score, Variable, Self::State)> {
        let state = if let Some(state) = state {
            state
        } else {
            self.initialize(problem)?
        };
        let (value, solution, state) = self.step(problem, state)?;
        Ok((value, solution, state))
    }
}
