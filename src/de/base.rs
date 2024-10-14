use anyhow::Result as AnyResult;

use crate::interface::{Problem, Score, Variable};

pub trait Initializer {
    fn initialize(
        &self,
        problem: &dyn Problem,
        population_size: usize,
    ) -> AnyResult<(Vec<Score>, Vec<Variable>)>;
}

pub trait MutationOperator {
    fn mutate_one(&self, current_population: &[Variable]) -> AnyResult<Variable>;
    fn mutate_all(&self, current_population: &[Variable]) -> AnyResult<Vec<Variable>> {
        let n = current_population.len();
        (0..n)
            .map(|_| self.mutate_one(current_population))
            .collect()
    }
}

pub trait CrossoverOperator {
    fn crossover_one(&self, v_current: &Variable, v_mutant: &Variable) -> AnyResult<Variable>;
    fn crossover_all(
        &self,
        current_population: &[Variable],
        mutant_population: &[Variable],
    ) -> AnyResult<Vec<Variable>> {
        current_population
            .iter()
            .zip(mutant_population)
            .map(|(current, mutant)| self.crossover_one(current, mutant))
            .collect()
    }
}

pub trait Selector {
    fn select_one(
        &self,
        problem: &dyn Problem,
        s_current: Score,
        v_current: Variable,
        v_trial: Variable,
    ) -> AnyResult<(Score, Variable)>;

    fn select_all(
        &self,
        problem: &dyn Problem,
        current_scores: Vec<Score>,
        current_population: Vec<Variable>,
        trial_population: Vec<Variable>,
    ) -> AnyResult<(Vec<Score>, Vec<Variable>)> {
        current_scores
            .into_iter()
            .zip(current_population)
            .zip(trial_population)
            .map(|((current_score, current), trial)| {
                self.select_one(problem, current_score, current, trial)
            })
            .collect()
    }
}
