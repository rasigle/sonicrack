"""Example: Polyphonic MIDI synthesizer playing chords.

Demonstrates the PolyphonicSynth playing multiple notes simultaneously.
"""

from src.engine import ADSREnvelope, Chain, ModulatedVolume, SineOscillator
from src.midi_io import PolyphonicSynth, midi_to_note_name


def create_voice():
    """Create a voice (oscillator + envelope)."""
    osc = SineOscillator(440, amplitude=0.2)
    env = ADSREnvelope(
        attack_duration=0.05,
        decay_duration=0.1,
        sustain_level=0.7,
        release_duration=0.3,
    )
    return Chain(osc, ModulatedVolume(env))


# Create polyphonic synth with 8 voices
synth = PolyphonicSynth(create_voice, max_voices=8)

print("Polyphonic Synthesizer Demo")
print("=" * 60)
print(f"Synth: {synth}\n")

# Test 1: Play a C major chord
print("Test 1: C Major Chord (C-E-G)")
print("-" * 60)

notes = [60, 64, 67]  # C, E, G
print(f"Playing notes: {[midi_to_note_name(n) for n in notes]}")

for note in notes:
    synth.note_on(note, 100)
    print(f"  Note ON: {midi_to_note_name(note)} ({note})")

print(f"Active voices: {synth.get_active_voice_count()}")
print(f"Free voices: {synth.get_free_voice_count()}")

# Generate audio
samples = synth.get_samples(2000)
print(f"Generated {len(samples)} samples, max amplitude: {samples.max():.3f}")

# Release chord
print("\nReleasing chord...")
for note in notes:
    synth.note_off(note)
    print(f"  Note OFF: {midi_to_note_name(note)}")

# Generate release tail
release_samples = synth.get_samples(5000)
print(f"Release tail: {len(release_samples)} samples, max: {release_samples.max():.3f}")

print(f"\nAfter release: {synth}")

# Test 2: Play many notes (test voice stealing)
print("\n\nTest 2: Voice Stealing (play 10 notes with 8 voices)")
print("-" * 60)

scale = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76]  # C major scale + more

for note in scale:
    synth.note_on(note, 100)
    print(
        f"  Note ON: {midi_to_note_name(note):4s} - Active: "
        f"{synth.get_active_voice_count()}, Free: {synth.get_free_voice_count()}"
    )

print(f"\n{synth}")

# Generate audio with all voices
full_samples = synth.get_samples(1000)
print(f"Mixed output: {len(full_samples)} samples, max: {full_samples.max():.3f}")

# Test 3: Chord progression
print("\n\nTest 3: Chord Progression (C -> Am -> F -> G)")
print("-" * 60)

synth.reset()

chords = [
    ("C Major", [60, 64, 67]),
    ("A minor", [57, 60, 64]),
    ("F Major", [53, 57, 60]),
    ("G Major", [55, 59, 62]),
]

for chord_name, chord_notes in chords:
    # Play chord
    print(f"\n{chord_name}: {[midi_to_note_name(n) for n in chord_notes]}")
    for note in chord_notes:
        synth.note_on(note, 100)

    samples = synth.get_samples(2000)
    print(f"  Audio: max amplitude {samples.max():.3f}")

    # Release chord
    for note in chord_notes:
        synth.note_off(note)

    release = synth.get_samples(2000)
    print(f"  Release: max amplitude {release.max():.3f}")

print("\n" + "=" * 60)
print("Demo complete! ✅")
print("\nPolyphonic synth features demonstrated:")
print("  ✓ Multiple notes simultaneously (chords)")
print("  ✓ Voice allocation and mixing")
print("  ✓ Voice stealing when voices exhausted")
print("  ✓ Proper release handling")
print("  ✓ Chord progressions")
