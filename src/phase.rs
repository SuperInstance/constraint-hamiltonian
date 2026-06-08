use serde::{Deserialize, Serialize};

/// A single point in phase space (q, p) at a given time with energy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasePoint {
    pub time: f64,
    pub positions: Vec<f64>,
    pub momenta: Vec<f64>,
    pub energy: f64,
}

/// A full phase portrait (trajectory through phase space).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasePortrait {
    pub trajectory: Vec<PhasePoint>,
    pub total_energy_drift: f64,
}

impl PhasePortrait {
    /// Number of recorded points.
    pub fn len(&self) -> usize {
        self.trajectory.len()
    }

    /// Whether the portrait is empty.
    pub fn is_empty(&self) -> bool {
        self.trajectory.is_empty()
    }

    /// Extract the time series.
    pub fn times(&self) -> Vec<f64> {
        self.trajectory.iter().map(|p| p.time).collect()
    }

    /// Extract energy over time.
    pub fn energies(&self) -> Vec<f64> {
        self.trajectory.iter().map(|p| p.energy).collect()
    }

    /// Extract position component `i` over time.
    pub fn position_series(&self, i: usize) -> Vec<f64> {
        self.trajectory.iter().map(|p| p.positions[i]).collect()
    }

    /// Extract momentum component `i` over time.
    pub fn momentum_series(&self, i: usize) -> Vec<f64> {
        self.trajectory.iter().map(|p| p.momenta[i]).collect()
    }

    /// Export as CSV string.
    pub fn to_csv(&self) -> String {
        let n = self.trajectory.first().map(|p| p.positions.len()).unwrap_or(0);
        let mut header = String::from("time");
        for i in 0..n {
            header.push_str(&format!(",q{i}"));
        }
        for i in 0..n {
            header.push_str(&format!(",p{i}"));
        }
        header.push_str(",energy\n");

        let mut rows = header;
        for point in &self.trajectory {
            rows.push_str(&format!("{}", point.time));
            for q in &point.positions {
                rows.push_str(&format!(",{q:.12}"));
            }
            for p in &point.momenta {
                rows.push_str(&format!(",{p:.12}"));
            }
            rows.push_str(&format!(",{:.12}\n", point.energy));
        }
        rows
    }

    /// Export as JSON string (requires serde serialization).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Sample every k-th point (downsample for visualization).
    pub fn sample(&self, k: usize) -> PhasePortrait {
        let sampled: Vec<_> = self
            .trajectory
            .iter()
            .enumerate()
            .filter(|(i, _)| i % k == 0)
            .map(|(_, p)| p.clone())
            .collect();
        PhasePortrait {
            trajectory: sampled,
            total_energy_drift: self.total_energy_drift,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_portrait() -> PhasePortrait {
        PhasePortrait {
            trajectory: vec![
                PhasePoint {
                    time: 0.0,
                    positions: vec![1.0],
                    momenta: vec![0.0],
                    energy: 0.5,
                },
                PhasePoint {
                    time: 0.01,
                    positions: vec![0.99],
                    momenta: vec![0.1],
                    energy: 0.5001,
                },
                PhasePoint {
                    time: 0.02,
                    positions: vec![0.98],
                    momenta: vec![0.2],
                    energy: 0.5002,
                },
            ],
            total_energy_drift: 0.0002,
        }
    }

    #[test]
    fn test_len_and_empty() {
        let p = make_portrait();
        assert_eq!(p.len(), 3);
        assert!(!p.is_empty());

        let empty = PhasePortrait {
            trajectory: vec![],
            total_energy_drift: 0.0,
        };
        assert!(empty.is_empty());
    }

    #[test]
    fn test_times_and_energies() {
        let p = make_portrait();
        assert_eq!(p.times(), vec![0.0, 0.01, 0.02]);
        let e = p.energies();
        assert!((e[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_position_momentum_series() {
        let p = make_portrait();
        assert_eq!(p.position_series(0), vec![1.0, 0.99, 0.98]);
        assert!((p.momentum_series(0)[2] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_csv_export() {
        let p = make_portrait();
        let csv = p.to_csv();
        assert!(csv.contains("time,q0,p0,energy"));
        assert!(csv.contains("0.000000000000"));
    }

    #[test]
    fn test_json_export() {
        let p = make_portrait();
        let json = p.to_json().unwrap();
        assert!(json.contains("\"time\""));
        assert!(json.contains("0.5"));
    }

    #[test]
    fn test_sample() {
        let p = make_portrait();
        let sampled = p.sample(2);
        assert_eq!(sampled.len(), 2); // indices 0 and 2
    }
}
