"""Example: Simple monophonic MIDI synthesizer.

This example demonstrates using MonophonicSynth to create a simple synthesizer
that responds to MIDI input or plays notes programmatically.

Requirements:
    pip install mido python-rtmidi sounddevice

Usage:
    python examples/midi_monophonic_synth.py
"""

import sys
import time
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("Warning: sounddevice not installed. Audio playback disabled.")
    print("Install with: pip install sounddevice")

try:
    from src.engine.midi import (
        MonophonicSynth,
        MIDIInput,
        midi_to_note_name,
    )
    from src.engine.midi.input import MIDO_AVAILABLE
    from src.engine import SineOscillator, ADSREnvelope, Chain
    from src.engine.modifier import ModulatedVolume
    from src.constants import DEFAULT_SAMPLE_RATE
except ImportError as e:
    print(f"Error: Could not import modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


# ANSI colors
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"


def create_voice():
    """Create a simple synthesizer voice.

    Returns a chain of: Oscillator -> ADSR Envelope -> Modulated Volume
    """
    # Create oscillator (will be set to correct frequency when note triggers)
    osc = SineOscillator(frequency=440, amplitude=0.3)

    # Create ADSR envelope
    env = ADSREnvelope(
        attack_duration=0.05,  # Fast attack
        decay_duration=0.1,  # Short decay
        sustain_level=0.7,  # 70% sustain
        release_duration=0.2,  # Quick release
        sample_rate=DEFAULT_SAMPLE_RATE,
    )

    # Chain them together with ModulatedVolume
    mod_vol = ModulatedVolume(env)
    chain = Chain(osc, mod_vol)

    return chain


def demo_programmatic():
    """Demo: Play notes programmatically (no MIDI input needed)."""
    print(
        f"\n{Colors.BOLD}{Colors.CYAN}=== Monophonic Synth - Programmatic Demo ==={Colors.ENDC}\n"
    )

    if sd is None:
        print(
            f"{Colors.RED}sounddevice not installed - skipping audio playback{Colors.ENDC}"
        )
        return

    # Create synth
    synth = MonophonicSynth(create_voice)

    # Play a simple melody: C-E-G-C (C major arpeggio)
    melody = [
        (60, 0.5),  # C4, 0.5 seconds
        (64, 0.5),  # E4, 0.5 seconds
        (67, 0.5),  # G4, 0.5 seconds
        (72, 0.5),  # C5, 0.5 seconds
    ]

    print(f"{Colors.BOLD}Playing melody...{Colors.ENDC}")
    print(f"Notes: {' → '.join(midi_to_note_name(n) for n, _ in melody)}\n")

    for note, duration in melody:
        note_name = midi_to_note_name(note)
        print(f"♪ {note_name}", end="", flush=True)

        # Trigger note
        synth.note_on(note, velocity=100)

        # Generate audio for note duration
        num_samples = int(duration * DEFAULT_SAMPLE_RATE)
        samples = synth.get_samples(num_samples)

        # Play audio
        sd.play(samples, DEFAULT_SAMPLE_RATE)
        sd.wait()

        # Release note
        synth.note_off(note)

        # Generate release tail
        release_samples = synth.get_samples(int(0.1 * DEFAULT_SAMPLE_RATE))
        sd.play(release_samples, DEFAULT_SAMPLE_RATE)
        sd.wait()

        print(" ✓")

    print(f"\n{Colors.GREEN}Melody complete!{Colors.ENDC}\n")


def demo_midi_input():
    """Demo: Use real MIDI input."""
    print(
        f"\n{Colors.BOLD}{Colors.CYAN}=== Monophonic Synth - MIDI Input Demo ==={Colors.ENDC}\n"
    )

    if not MIDO_AVAILABLE:
        print(f"{Colors.RED}mido not installed{Colors.ENDC}")
        print("Install with: pip install mido python-rtmidi")
        return

    if sd is None:
        print(f"{Colors.RED}sounddevice not installed{Colors.ENDC}")
        print("Install with: pip install sounddevice")
        return

    # List MIDI devices
    try:
        devices = MIDIInput.list_devices()
    except Exception as e:
        print(f"{Colors.RED}Error listing MIDI devices: {e}{Colors.ENDC}")
        return

    if not devices:
        print(f"{Colors.YELLOW}No MIDI devices found{Colors.ENDC}")
        print("Try the programmatic demo instead (option 1)")
        return

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

    # Create synth
    synth = MonophonicSynth(create_voice)

    # Audio stream for real-time playback
    _ = np.zeros(1024, dtype=np.float32)

    def audio_callback(outdata, frames, time_info, status):
        """Audio callback for real-time playback."""

        # Generate samples from synth
        samples = synth.get_samples(frames)

        # Output to audio
        if len(samples.shape) == 1:  # Mono
            outdata[:, 0] = samples
            outdata[:, 1] = samples
        else:  # Stereo
            outdata[:] = samples

    # MIDI message handler
    def on_midi_message(msg):
        """Handle incoming MIDI messages."""
        from src.engine.midi.messages import NoteOnMessage, NoteOffMessage

        if isinstance(msg, NoteOnMessage):
            note_name = midi_to_note_name(msg.note)
            vel_percent = int(msg.normalize_velocity() * 100)
            print(
                f"{Colors.GREEN}♪ {note_name:4s} ON  "
                f"(vel: {msg.velocity:3d} / {vel_percent:3d}%){Colors.ENDC}"
            )
            synth.note_on(msg.note, msg.velocity)

        elif isinstance(msg, NoteOffMessage):
            note_name = midi_to_note_name(msg.note)
            print(f"{Colors.RED}♪ {note_name:4s} OFF{Colors.ENDC}")
            synth.note_off(msg.note)

    print(
        f"\n{Colors.BOLD}Listening for MIDI input... (Press Ctrl+C to stop){Colors.ENDC}\n"
    )

    # Start audio and MIDI
    try:
        with sd.OutputStream(
            channels=2,
            samplerate=DEFAULT_SAMPLE_RATE,
            blocksize=1024,
            callback=audio_callback,
        ):
            with MIDIInput(selected_device) as midi:
                midi.start(on_midi_message)

                # Keep running
                while True:
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.BOLD}Stopped.{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")


def main():
    """Main menu."""
    print(f"\n{Colors.BOLD}Monophonic MIDI Synthesizer Demo{Colors.ENDC}")
    print("\nChoose a demo:")
    print("  1. Programmatic (play notes from code)")
    print("  2. MIDI Input (play from MIDI keyboard)")
    print("  0. Exit")
    print("\nChoice: ", end="")

    try:
        choice = input().strip()

        if choice == "1":
            demo_programmatic()
        elif choice == "2":
            demo_midi_input()
        elif choice == "0":
            print("Goodbye!")
        else:
            print(f"{Colors.RED}Invalid choice{Colors.ENDC}")

    except KeyboardInterrupt:
        print(f"\n\n{Colors.BOLD}Stopped.{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")


if __name__ == "__main__":
    main()
