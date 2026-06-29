"""Example: Real-time polyphonic MIDI keyboard synthesizer.

This example demonstrates using PolyphonicSynth with real MIDI keyboard input
and real-time audio playback. Play chords on your MIDI keyboard and hear them
through your speakers!

Requirements:
    pip install mido python-rtmidi sounddevice

Usage:
    python examples/polyphonic_midi_keyboard.py

Then play your MIDI keyboard - chords and multiple notes work!
Press Ctrl+C to exit.
"""

import sys
import time

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("Error: sounddevice not installed")
    print("Install with: pip install sounddevice")
    sys.exit(1)


from engine import (
    ADSREnvelope,
    Chain,
    ModulatedVolume,
    SawtoothOscillator,
    SineOscillator,
    SquareOscillator,
)
from midi_io import (
    MIDIInput,
    NoteOffMessage,
    NoteOnMessage,
    PolyphonicSynth,
    midi_to_note_name,
)
from src.constants import DEFAULT_SAMPLE_RATE


# ANSI colors
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"


def create_piano_voice():
    """Create a piano-like voice."""
    # Use sine wave with fast attack and medium release
    osc = SineOscillator(440, amplitude=0.25)
    env = ADSREnvelope(
        attack_duration=0.01,  # Very fast attack
        decay_duration=0.1,  # Quick decay
        sustain_level=0.6,  # Moderate sustain
        release_duration=0.4,  # Medium release
        sample_rate=DEFAULT_SAMPLE_RATE,
    )
    return Chain(osc, ModulatedVolume(env))


def create_organ_voice():
    """Create an organ-like voice."""
    # Use square wave with no attack and long sustain
    osc = SquareOscillator(440, amplitude=0.2)
    env = ADSREnvelope(
        attack_duration=0.0,  # Instant attack
        decay_duration=0.0,  # No decay
        sustain_level=1.0,  # Full sustain
        release_duration=0.1,  # Quick release
        sample_rate=DEFAULT_SAMPLE_RATE,
    )
    return Chain(osc, ModulatedVolume(env))


def create_string_voice():
    """Create a string-like voice."""
    # Use sawtooth with slow attack and long release
    osc = SawtoothOscillator(440, amplitude=0.2)
    env = ADSREnvelope(
        attack_duration=0.2,  # Slow attack
        decay_duration=0.3,  # Slow decay
        sustain_level=0.8,  # High sustain
        release_duration=0.8,  # Long release
        sample_rate=DEFAULT_SAMPLE_RATE,
    )
    return Chain(osc, ModulatedVolume(env))


def select_voice_factory():
    """Let user select voice type."""
    print(f"\n{Colors.BOLD}Select Voice Type:{Colors.ENDC}")
    print("  1. Piano (sine wave, fast attack)")
    print("  2. Organ (square wave, instant attack)")
    print("  3. Strings (sawtooth, slow attack)")
    print("\nChoice (1-3) [default: 1]: ", end="")

    choice = input().strip()

    if choice == "2":
        print(f"{Colors.CYAN}Using: Organ voice{Colors.ENDC}")
        return create_organ_voice
    elif choice == "3":
        print(f"{Colors.CYAN}Using: String voice{Colors.ENDC}")
        return create_string_voice
    else:
        print(f"{Colors.CYAN}Using: Piano voice{Colors.ENDC}")
        return create_piano_voice


def main():
    """Main function."""
    print(
        f"\n{Colors.BOLD}{Colors.CYAN}=== Polyphonic MIDI Keyboard Synthesizer ==="
        f"{Colors.ENDC}\n"
    )

    # List MIDI devices
    try:
        devices = MIDIInput.list_devices()
    except Exception as e:
        print(f"{Colors.RED}Error listing MIDI devices: {e}{Colors.ENDC}")
        return

    if not devices:
        print(f"{Colors.RED}No MIDI devices found!{Colors.ENDC}")
        print("\nMake sure a MIDI keyboard is connected.")
        sys.exit(1)

    print(f"Found {len(devices)} MIDI device(s):\n")
    for i, device in enumerate(devices):
        print(f"  {i + 1}. {device}")

    # Select device
    if len(devices) == 1:
        selected_device = devices[0]
        print(f"\nUsing: {Colors.BOLD}{selected_device}{Colors.ENDC}")
    else:
        print(f"\nSelect device (1-{len(devices)}): ", end="")
        try:
            choice = int(input())
            if 1 <= choice <= len(devices):
                selected_device = devices[choice - 1]
            else:
                print(f"{Colors.RED}Invalid choice{Colors.ENDC}")
                return
        except ValueError:
            print(f"{Colors.RED}Invalid input{Colors.ENDC}")
            return
        print(f"Using: {Colors.BOLD}{selected_device}{Colors.ENDC}")

    # Select voice type
    voice_factory = select_voice_factory()

    # Create polyphonic synth
    max_voices = 16  # Support up to 16 simultaneous notes
    synth = PolyphonicSynth(voice_factory, max_voices=max_voices)

    print(f"\n{Colors.BOLD}Synth Configuration:{Colors.ENDC}")
    print(f"  Max voices: {max_voices}")
    print(f"  Sample rate: {DEFAULT_SAMPLE_RATE} Hz")
    print(f"  Status: {synth}")

    # Statistics tracking
    stats = {
        "notes_played": 0,
        "notes_released": 0,
        "max_polyphony": 0,
    }

    # MIDI message handler
    def on_midi_message(msg):
        """Handle incoming MIDI messages."""
        if isinstance(msg, NoteOnMessage):
            note_name = midi_to_note_name(msg.note)
            vel_percent = int(msg.normalize_velocity() * 100)

            # Update stats
            stats["notes_played"] += 1
            active = synth.get_active_voice_count()
            if active > stats["max_polyphony"]:
                stats["max_polyphony"] = active

            print(
                f"{Colors.GREEN}♪ {note_name:4s} ON  "
                f"vel:{msg.velocity:3d} ({vel_percent:3d}%) "
                f"[{active}/{max_voices} voices]{Colors.ENDC}"
            )
            synth.note_on(msg.note, msg.velocity)

        elif isinstance(msg, NoteOffMessage):
            note_name = midi_to_note_name(msg.note)
            stats["notes_released"] += 1
            active = synth.get_active_voice_count()

            print(
                f"{Colors.RED}♪ {note_name:4s} OFF "
                f"[{active}/{max_voices} voices]{Colors.ENDC}"
            )
            synth.note_off(msg.note)

    # Audio callback for real-time playback
    def audio_callback(outdata, frames, time_info, status):
        """Audio callback for real-time playback."""
        if status:
            print(f"{Colors.YELLOW}Audio status: {status}{Colors.ENDC}")

        # Generate samples from synth
        samples = synth.get_samples(frames)

        # Output to audio (stereo)
        if len(samples.shape) == 1:  # Mono
            outdata[:, 0] = samples
            outdata[:, 1] = samples
        else:  # Stereo
            outdata[:] = samples

    print(f"\n{Colors.BOLD}{Colors.GREEN}Listening for MIDI input...{Colors.ENDC}")
    print(
        f"{Colors.BOLD}Play your MIDI keyboard! (Press Ctrl+C to stop){Colors.ENDC}\n"
    )
    print("=" * 70)

    # Start audio and MIDI
    try:
        with (
            sd.OutputStream(
                channels=2,
                samplerate=DEFAULT_SAMPLE_RATE,
                blocksize=1024,
                callback=audio_callback,
            ),
            MIDIInput(selected_device) as midi,
        ):
            midi.start(on_midi_message)

            # Keep running
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.BOLD}Stopped.{Colors.ENDC}")
        print("\n" + "=" * 70)
        print(f"{Colors.BOLD}Session Statistics:{Colors.ENDC}")
        print(f"  Notes played: {stats['notes_played']}")
        print(f"  Notes released: {stats['notes_released']}")
        print(f"  Max polyphony: {stats['max_polyphony']} voices")
        print(f"  Final state: {synth}")
        print("\nThank you for playing! 🎹")

    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
