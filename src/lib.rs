//! # constraint-hamiltonian
//!
//! Hamiltonian constraint systems on graphs with symplectic integration.
//!
//! This crate provides tools for simulating Hamiltonian dynamical systems with
//! holonomic constraints, using symplectic (energy-preserving) integrators.
//!
//! ## Core Concepts
//!
//! - **Hamiltonian**: H(q, p) = p^T M^{-1} p / 2 + V(q) — kinetic + potential energy
//! - **Symplectic integrators**: Verlet and Störmer-Verlet methods that preserve
//!   the symplectic structure of Hamiltonian mechanics
//! - **Constraints**: Holonomic constraints g(q) = 0 projected onto the tangent
//!   space using SHAKE/RATTLE-style projections
//!
//! ## Quick Start
//!
//! ```rust
//! use constraint_hamiltonian::{HamiltonianSystem, SymplecticIntegrator, IntegrationMethod};
//!
//! // Simple harmonic oscillator: H = p²/2 + q²/2
//! let mut system = HamiltonianSystem::new(
//!     vec![1.0],           // initial position
//!     vec![0.0],           // initial momentum
//!     vec![1.0],           // mass
//!     Box::new(|q| 0.5 * q[0] * q[0]), // potential V(q) = q²/2
//!     vec![],              // no constraints
//!     0.01,                // time step
//! ).unwrap();
//!
//! let mut integrator = SymplecticIntegrator::new(IntegrationMethod::Verlet, 1000);
//! let portrait = integrator.integrate(&mut system);
//!
//! // Verlet preserves energy well
//! assert!(integrator.max_energy_drift() < 0.001);
//! ```

pub mod conservation;
pub mod constraint;
pub mod error;
pub mod hamiltonian;
pub mod integrator;
pub mod midi;
pub mod phase;

// Re-export main types
pub use conservation::{check_energy_conservation, drift_report, DriftReport};
pub use constraint::Constraint;
pub use error::{HamiltonianError, Result};
pub use hamiltonian::{HamiltonianSystem, HamiltonianSystemBuilder};
pub use integrator::{IntegrationMethod, SymplecticIntegrator};
pub use midi::{export_midi, phase_to_midi, MidiEvent};
pub use phase::{PhasePoint, PhasePortrait};
