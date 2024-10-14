use anyhow::Result as AnyResult;

use super::base::{CrossoverOperator, Initializer, MutationOperator, Selector};

use crate::interface::{Optimizer, Problem, Score, Variable};

pub struct DEOptimizer<I: Initializer, M: MutationOperator, C: CrossoverOperator, S: Selector> {
    population_size: usize,
    initializer: I,
    mutation_operator: M,
    crossover_operator: C,
    selector: S,
}

impl<I: Initializer, M: MutationOperator, C: CrossoverOperator, S: Selector>
    DEOptimizer<I, M, C, S>
{
    pub fn new(
        population_size: usize,
        initializer: I,
        mutation_operator: M,
        crossover_operator: C,
        selector: S,
    ) -> Self {
        Self {
            population_size,
            initializer,
            mutation_operator,
            crossover_operator,
            selector,
        }
    }
}

impl<I: Initializer, M: MutationOperator, C: CrossoverOperator, S: Selector> Optimizer
    for DEOptimizer<I, M, C, S>
{
    type State = (Vec<Score>, Vec<Variable>);

    fn initialize(&self, problem: &dyn Problem) -> AnyResult<Self::State> {
        self.initializer.initialize(problem, self.population_size)
    }

    fn extract_best(
        &self,
        _problem: &dyn Problem,
        state: &Self::State,
    ) -> AnyResult<(Score, Variable)> {
        let best = state
            .0
            .iter()
            .zip(state.1.iter())
            .min_by_key(|(score, _)| *score)
            .map(|(score, variable)| (*score, variable.clone()))
            .unwrap();
        Ok(best)
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
