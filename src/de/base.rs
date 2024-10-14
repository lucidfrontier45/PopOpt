use anyhow::Result as AnyResult;

use crate::interface::{Optimizer, Problem, Score, Variable};

pub trait Initializer {
    fn initialize(&self, problem: &dyn Problem) -> AnyResult<(Vec<Score>, Vec<Variable>)>;
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

pub struct DifferentialEvolution<
    I: Initializer,
    M: MutationOperator,
    C: CrossoverOperator,
    S: Selector,
> {
    initializer: I,
    mutation_operator: M,
    crossover_operator: C,
    selector: S,
}

impl<I: Initializer, M: MutationOperator, C: CrossoverOperator, S: Selector>
    DifferentialEvolution<I, M, C, S>
{
    pub fn new(initializer: I, mutation_operator: M, crossover_operator: C, selector: S) -> Self {
        Self {
            initializer,
            mutation_operator,
            crossover_operator,
            selector,
        }
    }
}

impl<I: Initializer, M: MutationOperator, C: CrossoverOperator, S: Selector> Optimizer
    for DifferentialEvolution<I, M, C, S>
{
    type State = (Vec<Score>, Vec<Variable>);

    fn initialize(&self, problem: &dyn Problem) -> AnyResult<Self::State> {
        self.initializer.initialize(problem)
    }

    fn step(
        &self,
        problem: &dyn Problem,
        state: Self::State,
    ) -> AnyResult<(Score, Variable, Self::State)> {
        let (current_scores, current_population) = state;
        let (best_score, best_variable) = current_scores
            .iter()
            .zip(current_population.iter())
            .min_by_key(|(score, _)| *score)
            .map(|(score, variable)| (*score, variable.clone()))
            .unwrap();
        let mutant_polulation = self.mutation_operator.mutate_all(&current_population)?;
        let trial_polulation = self
            .crossover_operator
            .crossover_all(&current_population, &mutant_polulation)?;
        let selected = self.selector.select_all(
            problem,
            current_scores,
            current_population,
            trial_polulation,
        )?;

        let (&best_selected_score, best_selected_variable) = selected
            .0
            .iter()
            .zip(selected.1.iter())
            .min_by_key(|(score, _)| *score)
            .unwrap();

        let (best_score, best_variable) = if best_selected_score < best_score {
            (best_selected_score, best_selected_variable.clone())
        } else {
            (best_score, best_variable)
        };

        Ok((best_score, best_variable, (selected.0, selected.1)))
    }
}
