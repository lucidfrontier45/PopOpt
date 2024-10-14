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

    fn extract_best(
        &self,
        problem: &dyn Problem,
        state: &Self::State,
    ) -> AnyResult<(Score, Variable)>;

    fn step(
        &self,
        problem: &dyn Problem,
        state: Self::State,
    ) -> AnyResult<(Score, Variable, Self::State)>;

    fn optimize(
        &self,
        n_iter: usize,
        problem: &dyn Problem,
        state: Option<Self::State>,
    ) -> AnyResult<(Score, Variable, Self::State)> {
        let mut state = if let Some(state) = state {
            state
        } else {
            self.initialize(problem)?
        };
        let (mut best_score, mut best_variable) = self.extract_best(problem, &state)?;
        for _ in 0..n_iter {
            let (score, variable, new_state) = self.step(problem, state)?;
            state = new_state;
            if score < best_score {
                best_score = score;
                best_variable = variable;
            }
        }
        Ok((best_score, best_variable, state))
    }
}
