"""Example: Display MIDI messages from an input device.

This example shows how to receive and display MIDI messages in real-time
from a connected MIDI controller or keyboard.

Requirements:
    pip install mido python-rtmidi

Usage:
    python examples/midi_input_demo.py

Features:
    - Lists all available MIDI devices
    - Displays messages with timestamps
    - Color-coded output for different message types
    - Shows note names for note on/off messages
    - Displays CC controller names for common controllers

Press Ctrl+C to exit.
"""

import sys
import time


try:
    from engine.io.midi import MIDIInput, MIDO_AVAILABLE
    from engine.io.midi import (
        MIDIMessage,
        NoteOnMessage,
        NoteOffMessage,
        ControlChangeMessage,
        PitchBendMessage,
        ProgramChangeMessage,
        AftertouchMessage,
    )
    from engine.io.midi import midi_to_note_name, midi_to_frequency
except ImportError:
    print("Error: Could not import MIDI modules")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


# ANSI color codes for terminal output
class Colors:
    """Terminal color codes."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


# Common CC controller names
CC_NAMES = {
    1: "Mod Wheel",
    2: "Breath",
    4: "Foot Controller",
    5: "Portamento Time",
    7: "Volume",
    8: "Balance",
    10: "Pan",
    11: "Expression",
    64: "Sustain Pedal",
    65: "Portamento",
    66: "Sostenuto",
    67: "Soft Pedal",
    71: "Resonance",
    72: "Release Time",
    73: "Attack Time",
    74: "Cutoff",
    75: "Decay Time",
    91: "Reverb",
    93: "Chorus",
}


def format_message(msg: MIDIMessage) -> str:
    """Format a MIDI message for display.

    Args:
        msg: MIDI message to format

    Returns:
        Formatted string with color codes
    """
    timestamp_str = f"[{msg.timestamp:6.2f}s]"
    channel_str = f"Ch{msg.channel + 1:2d}"

    if isinstance(msg, NoteOnMessage):
        note_name = midi_to_note_name(msg.note)
        freq = midi_to_frequency(msg.note)
        vel_percent = int(msg.normalize_velocity() * 100)
        return (
            f"{Colors.GREEN}{timestamp_str} {channel_str} "
            f"NOTE ON  {Colors.BOLD}{note_name:4s}{Colors.ENDC}{Colors.GREEN} "
            f"(#{msg.note:3d}, {freq:7.2f}Hz) "
            f"Velocity: {msg.velocity:3d} ({vel_percent:3d}%){Colors.ENDC}"
        )

    elif isinstance(msg, NoteOffMessage):
        note_name = midi_to_note_name(msg.note)
        return (
            f"{Colors.RED}{timestamp_str} {channel_str} "
            f"NOTE OFF {Colors.BOLD}{note_name:4s}{Colors.ENDC}{Colors.RED} "
            f"(#{msg.note:3d}){Colors.ENDC}"
        )

    elif isinstance(msg, ControlChangeMessage):
        cc_name = CC_NAMES.get(msg.controller, f"CC{msg.controller}")
        value_percent = int(msg.normalize_value() * 100)

        # Special handling for switches
        if msg.controller == 64:  # Sustain pedal
            state = "ON " if msg.is_switch_on() else "OFF"
            return (
                f"{Colors.YELLOW}{timestamp_str} {channel_str} "
                f"CC       {cc_name:20s} = {Colors.BOLD}{state}{Colors.ENDC}"
            )
        else:
            return (
                f"{Colors.YELLOW}{timestamp_str} {channel_str} "
                f"CC       {cc_name:20s} = {msg.value:3d} ({value_percent:3d}%){Colors.ENDC}"
            )

    elif isinstance(msg, PitchBendMessage):
        semitones = msg.to_semitones(bend_range=2)
        direction = "↑" if msg.value > 0 else "↓" if msg.value < 0 else "•"
        return (
            f"{Colors.CYAN}{timestamp_str} {channel_str} "
            f"PITCHBND {direction} {msg.value:+6d} "
            f"({semitones:+.2f} semitones){Colors.ENDC}"
        )

    elif isinstance(msg, ProgramChangeMessage):
        return (
            f"{Colors.BLUE}{timestamp_str} {channel_str} "
            f"PROGRAM  → {msg.program}{Colors.ENDC}"
        )

    elif isinstance(msg, AftertouchMessage):
        pressure_percent = int(msg.normalize_pressure() * 100)
        return (
            f"{Colors.CYAN}{timestamp_str} {channel_str} "
            f"AFTERTCH Pressure: {msg.pressure:3d} ({pressure_percent:3d}%){Colors.ENDC}"
        )

    else:
        return f"{timestamp_str} {channel_str} {type(msg).__name__}"


def main():
    """Main function."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}=== MIDI Input Demo ==={Colors.ENDC}\n")

    # Check if mido is available
    if not MIDO_AVAILABLE:
        print(f"{Colors.RED}Error: mido library not installed{Colors.ENDC}")
        print("\nOr with uv:")
        print("  uv pip install mido python-rtmidi")
        sys.exit(1)

    # List available devices
    try:
        devices = MIDIInput.list_devices()
    except Exception as e:
        print(f"{Colors.RED}Error listing MIDI devices: {e}{Colors.ENDC}")
        sys.exit(1)

    if not devices:
        print(f"{Colors.RED}No MIDI input devices found!{Colors.ENDC}")
        print("\nMake sure a MIDI controller is connected.")
        sys.exit(1)

    print(f"Found {len(devices)} MIDI input device(s):\n")
    for i, device in enumerate(devices):
        print(f"  {i + 1}. {device}")

    # Select device
    if len(devices) == 1:
        selected_device = devices[0]
        print(f"\nUsing: {Colors.BOLD}{selected_device}{Colors.ENDC}")
    else:
        print(
            f"\nSelect device (1-{len(devices)}) or press Enter for first device: ",
            end="",
        )
        choice = input().strip()

        if choice == "":
            selected_device = devices[0]
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(devices):
                    selected_device = devices[idx]
                else:
                    print(f"{Colors.RED}Invalid choice{Colors.ENDC}")
                    sys.exit(1)
            except ValueError:
                print(f"{Colors.RED}Invalid input{Colors.ENDC}")
                sys.exit(1)

        print(f"Using: {Colors.BOLD}{selected_device}{Colors.ENDC}")

    # Show exit instructions
    print(
        f"\n{Colors.BOLD}Listening for MIDI messages... (Press Ctrl+C to exit){Colors.ENDC}\n"
    )
    print("=" * 80)

    # Create MIDI input with callback
    def on_message(msg: MIDIMessage):
        """Callback for received messages."""
        print(format_message(msg))

    # Start receiving
    try:
        with MIDIInput(selected_device) as midi:
            midi.start(on_message)

            # Keep running until interrupted
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.BOLD}Stopped.{Colors.ENDC}")
    except EOFError:
        # Ctrl+Z on Windows
        print(f"\n\n{Colors.BOLD}Stopped.{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
