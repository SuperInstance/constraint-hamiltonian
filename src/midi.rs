use crate::error::{HamiltonianError, Result};
use crate::phase::PhasePortrait;

/// Map position → MIDI pitch (0-127) and momentum → velocity (0-127).
pub fn phase_to_midi(portrait: &PhasePortrait) -> Vec<MidiEvent> {
    let mut events = Vec::with_capacity(portrait.trajectory.len());
    for point in &portrait.trajectory {
        for (i, &q) in point.positions.iter().enumerate() {
            let pitch = position_to_pitch(q);
            let velocity = momentum_to_velocity(
                point.momenta.get(i).copied().unwrap_or(0.0),
            );
            events.push(MidiEvent {
                time_ticks: (point.time * 480.0).round() as u32, // 480 ticks per beat
                pitch,
                velocity,
                channel: (i % 16) as u8,
            });
        }
    }
    events
}

/// Convert a position value to a MIDI pitch (0-127).
/// Uses a linear mapping with configurable range.
pub fn position_to_pitch(position: f64) -> u8 {
    // Map centered around middle C (60), range ±2 octaves
    let pitch = 60.0 + position * 12.0; // 1 unit = 1 octave
    pitch.clamp(0.0, 127.0).round() as u8
}

/// Convert a momentum value to a MIDI velocity (0-127).
pub fn momentum_to_velocity(momentum: f64) -> u8 {
    let vel = 64.0 + momentum * 32.0; // centered at 64
    vel.clamp(1.0, 127.0).round() as u8
}

/// A single MIDI note event.
#[derive(Debug, Clone, PartialEq)]
pub struct MidiEvent {
    pub time_ticks: u32,
    pub pitch: u8,
    pub velocity: u8,
    pub channel: u8,
}

/// Export MIDI events as a basic Standard MIDI File (Format 0).
pub fn export_midi(portrait: &PhasePortrait) -> Result<Vec<u8>> {
    let events = phase_to_midi(portrait);
    if events.is_empty() {
        return Err(HamiltonianError::MidiExport(
            "no events to export".into(),
        ));
    }

    let mut data = Vec::new();

    // Group events by time
    let mut timed_events: Vec<(u32, Vec<&MidiEvent>)> = Vec::new();
    let mut current_time = 0u32;
    let mut current_batch = Vec::new();

    // Sort events by time
    let mut sorted = events.clone();
    sorted.sort_by_key(|e| e.time_ticks);

    for event in &sorted {
        if event.time_ticks != current_time && !current_batch.is_empty() {
            timed_events.push((current_time, std::mem::take(&mut current_batch)));
            current_time = event.time_ticks;
        }
        current_batch.push(event);
    }
    if !current_batch.is_empty() {
        timed_events.push((current_time, current_batch));
    }

    // Build track data
    let mut track_data = Vec::new();

    // Tempo meta event: 120 BPM = 500000 microseconds per beat
    track_data.extend_from_slice(&[0x00]); // delta time
    track_data.extend_from_slice(&[0xFF, 0x51, 0x03]);
    track_data.extend_from_slice(&[0x07, 0xA1, 0x20]); // 500000 µs

    let mut last_time = 0u32;
    for (time, batch) in &timed_events {
        let delta = time.saturating_sub(last_time);
        write_variable_length(&mut track_data, delta);

        for event in batch {
            // Note on
            track_data.push(0x90 | (event.channel & 0x0F));
            track_data.push(event.pitch & 0x7F);
            track_data.push(event.velocity & 0x7F);
        }
        last_time = *time;

        // Note off after short duration
        write_variable_length(&mut track_data, 120); // ~quarter note
        for event in batch {
            track_data.push(0x80 | (event.channel & 0x0F));
            track_data.push(event.pitch & 0x7F);
            track_data.push(0x00);
        }
    }

    // End of track
    track_data.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

    // MIDI file header: MThd
    data.extend_from_slice(b"MThd");
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x06]); // header length
    data.extend_from_slice(&[0x00, 0x00]); // format 0
    data.extend_from_slice(&[0x00, 0x01]); // 1 track
    data.extend_from_slice(&[0x01, 0xE0]); // 480 ticks per beat

    // Track chunk: MTrk
    data.extend_from_slice(b"MTrk");
    let track_len = track_data.len() as u32;
    data.extend_from_slice(&track_len.to_be_bytes());
    data.extend_from_slice(&track_data);

    Ok(data)
}

/// Write a variable-length quantity (MIDI standard).
fn write_variable_length(data: &mut Vec<u8>, mut value: u32) {
    if value == 0 {
        data.push(0x00);
        return;
    }
    let mut bytes = Vec::new();
    while value > 0 {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if !bytes.is_empty() {
            byte |= 0x80;
        }
        bytes.push(byte);
    }
    bytes.reverse();
    data.extend_from_slice(&bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonian::HamiltonianSystem;
    use crate::integrator::{IntegrationMethod, SymplecticIntegrator};
    use crate::phase::PhasePoint;

    fn simple_portrait() -> PhasePortrait {
        PhasePortrait {
            trajectory: vec![
                PhasePoint {
                    time: 0.0,
                    positions: vec![0.0],
                    momenta: vec![1.0],
                    energy: 0.5,
                },
                PhasePoint {
                    time: 0.01,
                    positions: vec![1.0],
                    momenta: vec![0.0],
                    energy: 0.5,
                },
            ],
            total_energy_drift: 0.0,
        }
    }

    #[test]
    fn test_position_to_pitch() {
        assert_eq!(position_to_pitch(0.0), 60); // middle C
        assert_eq!(position_to_pitch(1.0), 72); // C5
        assert_eq!(position_to_pitch(-1.0), 48); // C3
    }

    #[test]
    fn test_momentum_to_velocity() {
        assert_eq!(momentum_to_velocity(0.0), 64);
        assert!(momentum_to_velocity(10.0) > 64);
        assert!(momentum_to_velocity(-10.0) < 64);
    }

    #[test]
    fn test_velocity_clamping() {
        let v = momentum_to_velocity(1000.0);
        assert_eq!(v, 127);
        let v2 = momentum_to_velocity(-1000.0);
        assert_eq!(v2, 1);
    }

    #[test]
    fn test_pitch_clamping() {
        assert_eq!(position_to_pitch(-100.0), 0);
        assert_eq!(position_to_pitch(100.0), 127);
    }

    #[test]
    fn test_phase_to_midi() {
        let portrait = simple_portrait();
        let events = phase_to_midi(&portrait);
        assert_eq!(events.len(), 2); // 2 time points × 1 DOF
        assert_eq!(events[0].pitch, 60);
    }

    #[test]
    fn test_export_midi_valid_header() {
        let portrait = simple_portrait();
        let data = export_midi(&portrait).unwrap();
        // Check MIDI header
        assert_eq!(&data[0..4], b"MThd");
        assert_eq!(&data[8..10], &[0x00, 0x00]); // format 0
        assert_eq!(&data[10..12], &[0x00, 0x01]); // 1 track
        // Check track header
        assert_eq!(&data[14..18], b"MTrk");
    }

    #[test]
    fn test_export_midi_empty_error() {
        let portrait = PhasePortrait {
            trajectory: vec![],
            total_energy_drift: 0.0,
        };
        assert!(export_midi(&portrait).is_err());
    }

    #[test]
    fn test_variable_length_encoding() {
        let mut data = Vec::new();
        write_variable_length(&mut data, 0);
        assert_eq!(data, vec![0x00]);

        data.clear();
        write_variable_length(&mut data, 127);
        assert_eq!(data, vec![0x7F]);

        data.clear();
        write_variable_length(&mut data, 128);
        assert_eq!(data, vec![0x81, 0x00]);
    }

    #[test]
    fn test_midi_from_integration() {
        let mut sys = HamiltonianSystem::new(
            vec![0.5],
            vec![1.0],
            vec![1.0],
            Box::new(|q| 0.5 * q[0] * q[0]),
            vec![],
            0.01,
        )
        .unwrap();

        let mut integrator = SymplecticIntegrator::new(IntegrationMethod::Verlet, 50);
        let portrait = integrator.integrate(&mut sys);
        let data = export_midi(&portrait).unwrap();
        assert!(data.len() > 22); // At least header + some track data
    }
}
