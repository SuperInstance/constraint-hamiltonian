/// A scalar constraint g(q) = 0 on the configuration manifold.
#[allow(clippy::type_complexity)]
pub struct Constraint {
    pub name: String,
    pub evaluate: Box<dyn Fn(&[f64]) -> f64>,
    pub gradient: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    pub tolerance: f64,
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constraint")
            .field("name", &self.name)
            .field("tolerance", &self.tolerance)
            .finish()
    }
}

impl Constraint {
    /// Create a new constraint.
    #[allow(clippy::type_complexity)]
    pub fn new(
        name: impl Into<String>,
        evaluate: Box<dyn Fn(&[f64]) -> f64>,
        gradient: Box<dyn Fn(&[f64]) -> Vec<f64>>,
        tolerance: f64,
    ) -> Self {
        Self {
            name: name.into(),
            evaluate,
            gradient,
            tolerance,
        }
    }

    /// Evaluate the constraint: g(q) should be 0.
    pub fn evaluate(&self, positions: &[f64]) -> f64 {
        (self.evaluate)(positions)
    }

    /// Compute the gradient ∇g(q).
    pub fn gradient(&self, positions: &[f64]) -> Vec<f64> {
        (self.gradient)(positions)
    }

    /// Check if the constraint is satisfied within tolerance.
    pub fn is_satisfied(&self, positions: &[f64]) -> bool {
        self.evaluate(positions).abs() < self.tolerance
    }
}

/// Project momenta onto the tangent space of all constraints.
///
/// p -= G^T (G G^T)^{-1} G p
pub fn project_momenta(
    positions: &[f64],
    momenta: &mut [f64],
    constraints: &[Constraint],
) {
    if constraints.is_empty() {
        return;
    }

    let _n = positions.len();
    let _m = constraints.len();

    let jacobian: Vec<Vec<f64>> = constraints.iter().map(|c| c.gradient(positions)).collect();

    // G p
    let gp: Vec<f64> = jacobian
        .iter()
        .map(|row| row.iter().zip(momenta.iter()).map(|(g, p)| g * p).sum())
        .collect();

    // G G^T
    let ggt: Vec<Vec<f64>> = jacobian
        .iter()
        .map(|row_i| {
            jacobian
                .iter()
                .map(|row_j| row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum())
                .collect()
        })
        .collect();

    let lambda = solve_linear_system(&ggt, &gp);

    // p -= G^T lambda
    for (j, mom) in momenta.iter_mut().enumerate() {
        for (i, lam) in lambda.iter().enumerate() {
            *mom -= jacobian[i][j] * lam;
        }
    }
}

/// Project positions back onto the constraint manifold using Newton iteration.
pub fn project_positions(
    positions: &mut [f64],
    constraints: &[Constraint],
    max_iterations: usize,
) {
    for _ in 0..max_iterations {
        let converged = constraints
            .iter()
            .all(|c| c.evaluate(positions).abs() <= c.tolerance * 0.1);
        if converged {
            return;
        }

        let _n = positions.len();
        let _m = constraints.len();

        let jacobian: Vec<Vec<f64>> = constraints.iter().map(|c| c.gradient(positions)).collect();
        let values: Vec<f64> = constraints.iter().map(|c| c.evaluate(positions)).collect();

        let ggt: Vec<Vec<f64>> = jacobian
            .iter()
            .map(|row_i| {
                jacobian
                    .iter()
                    .map(|row_j| row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum())
                    .collect()
            })
            .collect();

        let lambda = solve_linear_system(&ggt, &values);

        for (j, pos) in positions.iter_mut().enumerate() {
            for (i, lam) in lambda.iter().enumerate() {
                *pos -= jacobian[i][j] * lam;
            }
        }
    }
}

/// Simple Gaussian elimination for a small linear system Ax = b.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut aug = vec![vec![0.0f64; n + 1]; n];
    for (i, aug_row) in aug.iter_mut().enumerate() {
        aug_row[..n].copy_from_slice(&a[i]);
        aug_row[n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let max_row = (col..n)
            .max_by(|&r1, &r2| aug[r1][col].abs().partial_cmp(&aug[r2][col].abs()).unwrap())
            .unwrap_or(col);
        aug.swap(col, max_row);

        if aug[col][col].abs() < 1e-15 {
            continue;
        }

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            let col_vals: Vec<f64> = aug[col][col..=n].to_vec();
            for (j, cv) in (col..=n).zip(col_vals.iter()) {
                aug[row][j] -= factor * cv;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < 1e-15 {
            continue;
        }
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_creation_and_evaluation() {
        let c = Constraint::new(
            "x_squared",
            Box::new(|q| q[0] * q[0] + q[1] * q[1] - 1.0),
            Box::new(|q| vec![2.0 * q[0], 2.0 * q[1]]),
            1e-8,
        );
        let q = vec![0.6, 0.8];
        assert!((c.evaluate(&q)).abs() < 1e-10);
        let grad = c.gradient(&q);
        assert!((grad[0] - 1.2).abs() < 1e-10);
        assert!((grad[1] - 1.6).abs() < 1e-10);
        assert!(c.is_satisfied(&q));
    }

    #[test]
    fn test_constraint_not_satisfied() {
        let c = Constraint::new(
            "unit",
            Box::new(|q| q[0] - 1.0),
            Box::new(|_q| vec![1.0]),
            1e-8,
        );
        assert!(!c.is_satisfied(&[0.5]));
    }

    #[test]
    fn test_project_momenta_sphere_constraint() {
        let c = Constraint::new(
            "sphere",
            Box::new(|q| q[0] * q[0] + q[1] * q[1] + q[2] * q[2] - 1.0),
            Box::new(|q| vec![2.0 * q[0], 2.0 * q[1], 2.0 * q[2]]),
            1e-10,
        );

        let positions = vec![1.0, 0.0, 0.0];
        let mut momenta = vec![1.0, 1.0, 0.0];
        project_momenta(&positions, &mut momenta, &[c]);

        assert!(momenta[0].abs() < 1e-10, "radial component should be zero: got {}", momenta[0]);
        assert!((momenta[1] - 1.0).abs() < 1e-10);
        assert!((momenta[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_project_momenta_no_constraints() {
        let mut momenta = vec![1.0, 2.0, 3.0];
        project_momenta(&[0.0, 0.0, 0.0], &mut momenta, &[]);
        assert_eq!(momenta, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_project_positions_onto_sphere() {
        let c = Constraint::new(
            "sphere",
            Box::new(|q| q[0] * q[0] + q[1] * q[1] + q[2] * q[2] - 1.0),
            Box::new(|q| vec![2.0 * q[0], 2.0 * q[1], 2.0 * q[2]]),
            1e-10,
        );

        let mut positions = vec![1.1, 0.0, 0.0];
        project_positions(&mut positions, &[c], 20);

        let r2 = positions[0] * positions[0] + positions[1] * positions[1] + positions[2] * positions[2];
        assert!((r2 - 1.0).abs() < 1e-6, "should be on sphere: r^2 = {r2}");
    }

    #[test]
    fn test_solve_linear_system_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0, 4.0];
        let x = solve_linear_system(&a, &b);
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_system_general() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 7.0];
        let x = solve_linear_system(&a, &b);
        assert!((x[0] - 1.6).abs() < 1e-10);
        assert!((x[1] - 1.8).abs() < 1e-10);
    }
}
