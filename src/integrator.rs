use crate::constraint::{project_momenta, project_positions};
use crate::hamiltonian::HamiltonianSystem;
use crate::phase::{PhasePoint, PhasePortrait};
use serde::{Deserialize, Serialize};

/// Integration method.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum IntegrationMethod {
    /// Forward Euler (non-symplectic baseline).
    Euler,
    /// Velocity Verlet (symplectic, 2nd order).
    Verlet,
    /// Störmer-Verlet / leapfrog (symplectic, 2nd order).
    StormerVerlet,
}

/// Symplectic integrator with energy tracking.
#[derive(Debug)]
pub struct SymplecticIntegrator {
    pub method: IntegrationMethod,
    pub steps: usize,
    pub energy_drift: Vec<f64>,
}

impl SymplecticIntegrator {
    pub fn new(method: IntegrationMethod, steps: usize) -> Self {
        Self {
            method,
            steps,
            energy_drift: Vec::with_capacity(steps),
        }
    }

    /// Integrate the system, returning a phase portrait.
    pub fn integrate(&mut self, system: &mut HamiltonianSystem) -> PhasePortrait {
        let initial_energy = system.total_energy();
        let dt = system.dt;

        let mut trajectory = Vec::with_capacity(self.steps + 1);
        trajectory.push(PhasePoint {
            time: 0.0,
            positions: system.positions.clone(),
            momenta: system.momenta.clone(),
            energy: initial_energy,
        });

        for step in 1..=self.steps {
            match self.method {
                IntegrationMethod::Euler => self.step_euler(system, dt),
                IntegrationMethod::Verlet => self.step_verlet(system, dt),
                IntegrationMethod::StormerVerlet => self.step_stormer_verlet(system, dt),
            }

            // Project constraints
            if !system.constraints.is_empty() {
                project_momenta(&system.positions, &mut system.momenta, &system.constraints);
                project_positions(&mut system.positions, &system.constraints, 10);
            }

            let energy = system.total_energy();
            self.energy_drift.push(energy - initial_energy);

            trajectory.push(PhasePoint {
                time: step as f64 * dt,
                positions: system.positions.clone(),
                momenta: system.momenta.clone(),
                energy,
            });
        }

        let total_drift = self.energy_drift.iter().last().copied().unwrap_or(0.0);

        PhasePortrait {
            trajectory,
            total_energy_drift: total_drift,
        }
    }

    /// Forward Euler step: non-symplectic baseline.
    fn step_euler(&self, system: &mut HamiltonianSystem, dt: f64) {
        let n = system.dof();
        let forces = system.forces();

        // q_{n+1} = q_n + dt * M^{-1} p_n
        // p_{n+1} = p_n + dt * F_n
        let mut new_positions = vec![0.0; n];
        let mut new_momenta = vec![0.0; n];
        for i in 0..n {
            new_positions[i] = system.positions[i] + dt * system.momenta[i] / system.mass_matrix[i];
            new_momenta[i] = system.momenta[i] + dt * forces[i];
        }
        system.positions = new_positions;
        system.momenta = new_momenta;
    }

    /// Velocity Verlet step: symplectic, 2nd order.
    /// 1) p_{n+1/2} = p_n + (dt/2) * F(q_n)
    /// 2) q_{n+1} = q_n + dt * M^{-1} p_{n+1/2}
    /// 3) p_{n+1} = p_{n+1/2} + (dt/2) * F(q_{n+1})
    fn step_verlet(&self, system: &mut HamiltonianSystem, dt: f64) {
        // Step 1: half-step momenta
        let forces = system.forces();
        for (mom, f) in system.momenta.iter_mut().zip(forces.iter()) {
            *mom += 0.5 * dt * f;
        }

        // Step 2: full-step positions
        for ((pos, mom), mass) in system.positions.iter_mut().zip(system.momenta.iter()).zip(system.mass_matrix.iter()) {
            *pos += dt * mom / mass;
        }

        // Step 3: half-step momenta with new forces
        let forces_new = system.forces();
        for (mom, f) in system.momenta.iter_mut().zip(forces_new.iter()) {
            *mom += 0.5 * dt * f;
        }
    }

    /// Störmer-Verlet / leapfrog step.
    /// Equivalent to velocity Verlet but formulated as:
    /// q_{n+1} = q_n + dt * M^{-1} p_n + (dt^2/2) * M^{-1} F_n
    /// p_{n+1} = p_n + (dt/2) * (F_n + F_{n+1})
    fn step_stormer_verlet(&self, system: &mut HamiltonianSystem, dt: f64) {
        let forces = system.forces();

        // Update positions
        for ((pos, mom), (mass, f)) in system.positions.iter_mut().zip(system.momenta.iter()).zip(system.mass_matrix.iter().zip(forces.iter())) {
            *pos += dt * mom / mass + 0.5 * dt * dt * f / mass;
        }

        // Compute new forces
        let forces_new = system.forces();

        // Update momenta
        for (mom, (f_old, f_new)) in system.momenta.iter_mut().zip(forces.iter().zip(forces_new.iter())) {
            *mom += 0.5 * dt * (f_old + f_new);
        }
    }

    /// Compute the maximum absolute energy drift over the trajectory.
    pub fn max_energy_drift(&self) -> f64 {
        self.energy_drift
            .iter()
            .map(|d| d.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Compute the RMS energy drift.
    pub fn rms_energy_drift(&self) -> f64 {
        if self.energy_drift.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = self.energy_drift.iter().map(|d| d * d).sum();
        (sum_sq / self.energy_drift.len() as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harmonic_system() -> HamiltonianSystem {
        HamiltonianSystem::new(
            vec![1.0],
            vec![0.0],
            vec![1.0],
            Box::new(|q| 0.5 * q[0] * q[0]),
            vec![],
            0.01,
        )
        .unwrap()
    }

    #[test]
    fn test_euler_drifts_energy() {
        let mut sys = harmonic_system();
        let mut integrator = SymplecticIntegrator::new(IntegrationMethod::Euler, 1000);
        let _portrait = integrator.integrate(&mut sys);
        // Euler should show noticeable drift
        assert!(integrator.max_energy_drift() > 0.001);
    }

    #[test]
    fn test_verlet_preserves_energy() {
        let mut sys = harmonic_system();
        let mut integrator = SymplecticIntegrator::new(IntegrationMethod::Verlet, 1000);
        let _portrait = integrator.integrate(&mut sys);
        assert!(
            integrator.max_energy_drift() < 0.001,
            "Verlet energy drift too large: {}",
            integrator.max_energy_drift()
        );
    }

    #[test]
    fn test_stormer_verlet_preserves_energy() {
        let mut sys = harmonic_system();
        let mut integrator = SymplecticIntegrator::new(IntegrationMethod::StormerVerlet, 1000);
        let _portrait = integrator.integrate(&mut sys);
        assert!(
            integrator.max_energy_drift() < 0.001,
            "Störmer-Verlet energy drift too large: {}",
            integrator.max_energy_drift()
        );
    }

    #[test]
    fn test_verlet_better_than_euler() {
        let mut sys_euler = harmonic_system();
        let mut sys_verlet = harmonic_system();

        let mut euler = SymplecticIntegrator::new(IntegrationMethod::Euler, 1000);
        let mut verlet = SymplecticIntegrator::new(IntegrationMethod::Verlet, 1000);

        euler.integrate(&mut sys_euler);
        verlet.integrate(&mut sys_verlet);

        assert!(
            verlet.max_energy_drift() < euler.max_energy_drift(),
            "Verlet drift ({}) should be less than Euler drift ({})",
            verlet.max_energy_drift(),
            euler.max_energy_drift()
        );
    }

    #[test]
    fn test_harmonic_oscillator_frequency() {
        // For H = p^2/2 + q^2/2, frequency = 1, period = 2π
        let mut sys = HamiltonianSystem::new(
            vec![1.0],
            vec![0.0],
            vec![1.0],
            Box::new(|q| 0.5 * q[0] * q[0]),
            vec![],
            0.001, // Small dt for accuracy
        )
        .unwrap();

        let steps = 6283; // ~2π/0.001 ≈ one full period
        let mut integrator = SymplecticIntegrator::new(IntegrationMethod::Verlet, steps);
        let portrait = integrator.integrate(&mut sys);

        // After one period, should return close to initial position
        let final_q = portrait.trajectory.last().unwrap().positions[0];
        assert!(
            (final_q - 1.0).abs() < 0.01,
            "after one period q should be ~1.0, got {final_q}"
        );
    }

    #[test]
    fn test_phase_portrait_has_correct_length() {
        let mut sys = harmonic_system();
        let mut integrator = SymplecticIntegrator::new(IntegrationMethod::Verlet, 100);
        let portrait = integrator.integrate(&mut sys);
        assert_eq!(portrait.trajectory.len(), 101); // initial + 100 steps
    }

    #[test]
    fn test_constrained_particle_on_sphere() {

        let constraint = crate::Constraint::new(
            "sphere",
            Box::new(|q| q[0] * q[0] + q[1] * q[1] + q[2] * q[2] - 1.0),
            Box::new(|q| vec![2.0 * q[0], 2.0 * q[1], 2.0 * q[2]]),
            1e-6,
        );

        let mut sys = HamiltonianSystem::new(
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0], // tangential momentum
            vec![1.0, 1.0, 1.0],
            Box::new(|_| 0.0), // free particle on sphere
            vec![constraint],
            0.001,
        )
        .unwrap();

        let mut integrator = SymplecticIntegrator::new(IntegrationMethod::Verlet, 1000);
        let portrait = integrator.integrate(&mut sys);

        // Check constraint maintained at every step
        for point in &portrait.trajectory {
            let r2 = point.positions[0] * point.positions[0]
                + point.positions[1] * point.positions[1]
                + point.positions[2] * point.positions[2];
            assert!(
                (r2 - 1.0).abs() < 0.01,
                "constraint violated: |r^2 - 1| = {}",
                (r2 - 1.0).abs()
            );
        }
    }

    #[test]
    fn test_rms_drift() {
        let mut sys = harmonic_system();
        let mut integrator = SymplecticIntegrator::new(IntegrationMethod::Verlet, 100);
        integrator.integrate(&mut sys);
        let rms = integrator.rms_energy_drift();
        assert!(rms >= 0.0);
    }

    #[test]
    fn test_double_pendulum_energy_bounded() {
        // Double pendulum: 2 DOF, highly nonlinear potential
        let mut sys = HamiltonianSystem::new(
            vec![2.0, 1.0], // angles
            vec![0.0, 0.5], // angular momenta
            vec![1.0, 1.0],
            Box::new(|q| {
                // V = -cos(q1) - cos(q1 + q2) (gravitational)
                -q[0].cos() - (q[0] + q[1]).cos()
            }),
            vec![],
            0.001,
        )
        .unwrap();

        let _initial_energy = sys.total_energy();
        let mut integrator = SymplecticIntegrator::new(IntegrationMethod::Verlet, 10000);
        integrator.integrate(&mut sys);

        // Energy should stay bounded (Verlet is symplectic)
        assert!(
            integrator.max_energy_drift() < 0.1,
            "double pendulum energy drift too large: {}",
            integrator.max_energy_drift()
        );
    }
}
