use std::fmt;

/// Errors produced by the constraint-hamiltonian crate.
#[derive(Debug)]
pub enum HamiltonianError {
    /// Dimension mismatch between positions, momenta, or mass matrix.
    DimensionMismatch {
        expected: usize,
        found: usize,
        context: String,
    },
    /// Constraint violation exceeded tolerance.
    ConstraintViolation {
        name: String,
        value: f64,
        tolerance: f64,
    },
    /// Non-positive mass encountered.
    NonPositiveMass { index: usize, value: f64 },
    /// Non-positive time step.
    NonPositiveTimeStep(f64),
    /// Empty system (zero degrees of freedom).
    EmptySystem,
    /// MIDI export error.
    MidiExport(String),
    /// I/O error.
    Io(std::io::Error),
}

impl fmt::Display for HamiltonianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                expected,
                found,
                context,
            } => write!(
                f,
                "dimension mismatch in {context}: expected {expected}, found {found}"
            ),
            Self::ConstraintViolation {
                name,
                value,
                tolerance,
            } => write!(
                f,
                "constraint \"{name}\" violated: {value:.6e} exceeds tolerance {tolerance:.6e}"
            ),
            Self::NonPositiveMass { index, value } => {
                write!(f, "non-positive mass at index {index}: {value}")
            }
            Self::NonPositiveTimeStep(dt) => write!(f, "non-positive time step: {dt}"),
            Self::EmptySystem => write!(f, "system has zero degrees of freedom"),
            Self::MidiExport(msg) => write!(f, "MIDI export error: {msg}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for HamiltonianError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for HamiltonianError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, HamiltonianError>;
