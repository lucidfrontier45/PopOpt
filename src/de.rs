pub mod base;
pub mod optimizer;
pub mod simple;

use optimizer::DEOptimizer;
use ordered_float::NotNan;
use simple::{SimpleCrossoverOperator, SimpleInitializer, SimpleMutationOperator, SimpleSelector};

pub type SimpleDEOptimizer =
    DEOptimizer<SimpleInitializer, SimpleMutationOperator, SimpleCrossoverOperator, SimpleSelector>;

pub fn create_simple_de_optimizer(
    population_size: usize,
    bounds: Vec<(NotNan<f64>, NotNan<f64>)>,
    mutation_scale: NotNan<f64>,
    crossover_rate: NotNan<f64>,
) -> SimpleDEOptimizer {
    let initializer = SimpleInitializer::new(bounds);
    let mutation_operator = SimpleMutationOperator::new(mutation_scale);
    let crossover_operator = SimpleCrossoverOperator::new(crossover_rate);
    let selector = SimpleSelector::new();
    DEOptimizer::new(
        population_size,
        initializer,
        mutation_operator,
        crossover_operator,
        selector,
    )
}

#[cfg(test)]
mod tests {
    use anyhow::Result as AnyResult;
    use ordered_float::NotNan;

    use crate::interface::{Optimizer, Problem, Score, Variable};

    use super::create_simple_de_optimizer;

    fn rosenbrock(x: &Variable) -> Score {
        x.iter()
            .zip(x.iter().skip(1))
            .map(|(&x1, &x2)| {
                NotNan::new(100.0 * (x2 - x1 * x1).powi(2) + (1.0 - x1).powi(2)).unwrap()
            })
            .sum()
    }

    struct RosenbrockProblem;
    impl Problem for RosenbrockProblem {
        fn variable_dimension(&self) -> usize {
            2
        }

        fn evaluate(&self, x: &Variable) -> AnyResult<Score> {
            Ok(rosenbrock(x))
        }
    }

    #[test]
    fn test_de() {
        let bounds = vec![(NotNan::new(-5.0).unwrap(), NotNan::new(5.0).unwrap()); 2];
        let optimizer = create_simple_de_optimizer(
            20,
            bounds,
            NotNan::new(0.3).unwrap(),
            NotNan::new(0.3).unwrap(),
        );
        let problem = RosenbrockProblem;
        let (score, solution, _) = optimizer.optimize(500, &problem, None).unwrap();
        assert!(score.into_inner() < 1e-3);
        assert!((solution[0] - 1.0).abs() < 1e-2);
        assert!((solution[1] - 1.0).abs() < 1e-2);
    }
}
