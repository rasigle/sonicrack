"""Example: Analyze and display MIDI file contents.

This example shows how to load and analyze MIDI files using the MIDIFile class.

Requirements:
    pip install mido

Usage:
    python examples/midi_file_demo.py [path/to/file.mid]
    
If no file is provided, shows instructions on finding MIDI files.
"""

import sys
from pathlib import Path

try:
    from src.engine.midi import MIDIFile, midi_to_note_name
    from src.engine.midi.messages import (
        NoteOnMessage,
        NoteOffMessage,
        ControlChangeMessage,
        PitchBendMessage,
    )
    from src.engine.midi.file_reader import MIDO_AVAILABLE
except ImportError:
    print("Error: Could not import MIDI modules")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


# ANSI color codes
class Colors:
    """Terminal color codes."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def format_time(seconds: float) -> str:
    """Format time in MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string like "03:45"
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def analyze_midi_file(filepath: str):
    """Analyze and display MIDI file information.
    
    Args:
        filepath: Path to MIDI file
    """
    print(f"\n{Colors.BOLD}{Colors.HEADER}=== MIDI File Analyzer ==={Colors.ENDC}\n")
    
    # Load file
    try:
        midi = MIDIFile(filepath)
    except Exception as e:
        print(f"{Colors.RED}Error loading file: {e}{Colors.ENDC}")
        return
    
    print(f"{Colors.BOLD}File:{Colors.ENDC} {Path(filepath).name}")
    print("=" * 80)
    
    # Basic info
    print(f"\n{Colors.BOLD}Basic Information:{Colors.ENDC}")
    print(f"  Duration:        {format_time(midi.duration)} ({midi.duration:.2f} seconds)")
    print(f"  Tempo:           {midi.get_tempo():.1f} BPM")
    print(f"  Tracks:          {midi.get_track_count()}")
    print(f"  Total Messages:  {len(midi.messages)}")
    print(f"  Ticks/Beat:      {midi.ticks_per_beat}")
    
    # Message types
    print(f"\n{Colors.BOLD}Message Types:{Colors.ENDC}")
    message_types = midi.get_message_types()
    for msg_type, count in sorted(message_types.items()):
        print(f"  {msg_type:25s}: {count:5d}")
    
    # Note information
    note_count = midi.get_note_count()
    if note_count > 0:
        low_note, high_note = midi.get_note_range()
        low_name = midi_to_note_name(low_note)
        high_name = midi_to_note_name(high_note)
        
        print(f"\n{Colors.BOLD}Note Information:{Colors.ENDC}")
        print(f"  Total Notes:     {note_count}")
        print(f"  Note Range:      {low_name} ({low_note}) to {high_name} ({high_note})")
        print(f"  Range Span:      {high_note - low_note + 1} semitones")
    
    # Channel information
    channels = midi.get_used_channels()
    print(f"\n{Colors.BOLD}Channels:{Colors.ENDC}")
    print(f"  Active Channels: {len(channels)}")
    print(f"  Channel Numbers: {', '.join(str(ch + 1) for ch in channels)}")
    
    # Per-channel breakdown
    for channel in channels:
        ch_messages = midi.get_channel_messages(channel)
        ch_notes = sum(
            1 for msg in ch_messages
            if isinstance(msg, (NoteOnMessage, NoteOffMessage))
        )
        print(f"    Channel {channel + 1:2d}:    {len(ch_messages):5d} messages ({ch_notes} notes)")
    
    # First 10 messages preview
    print(f"\n{Colors.BOLD}First 10 Messages:{Colors.ENDC}")
    for i, msg in enumerate(midi.messages[:10]):
        timestamp_str = f"[{msg.timestamp:6.2f}s]"
        channel_str = f"Ch{msg.channel + 1:2d}"
        
        if isinstance(msg, NoteOnMessage):
            note_name = midi_to_note_name(msg.note)
            vel_percent = int(msg.normalize_velocity() * 100)
            info = f"NOTE ON  {note_name:4s} Vel:{msg.velocity:3d} ({vel_percent:3d}%)"
            color = Colors.GREEN
        elif isinstance(msg, NoteOffMessage):
            note_name = midi_to_note_name(msg.note)
            info = f"NOTE OFF {note_name:4s}"
            color = Colors.RED
        elif isinstance(msg, ControlChangeMessage):
            info = f"CC {msg.controller:3d} = {msg.value:3d}"
            color = Colors.YELLOW
        elif isinstance(msg, PitchBendMessage):
            info = f"PITCHBND = {msg.value:+6d}"
            color = Colors.CYAN
        else:
            info = type(msg).__name__
            color = Colors.BLUE
        
        print(f"  {color}{timestamp_str} {channel_str} {info}{Colors.ENDC}")
    
    if len(midi.messages) > 10:
        print(f"  {Colors.CYAN}... and {len(midi.messages) - 10} more messages{Colors.ENDC}")
    
    # Timeline analysis (every 10 seconds)
    print(f"\n{Colors.BOLD}Timeline (10-second intervals):{Colors.ENDC}")
    interval = 10.0
    current_time = 0.0
    
    while current_time < midi.duration:
        end_time = min(current_time + interval, midi.duration)
        messages = midi.get_notes_in_range(current_time, end_time)
        note_msgs = sum(
            1 for msg in messages
            if isinstance(msg, (NoteOnMessage, NoteOffMessage))
        )
        
        print(
            f"  {format_time(current_time)}-{format_time(end_time)}: "
            f"{len(messages):4d} messages ({note_msgs} notes)"
        )
        
        current_time += interval
    
    print("\n" + "=" * 80)


def main():
    """Main function."""
    # Check if mido is available
    if not MIDO_AVAILABLE:
        print(f"{Colors.RED}Error: mido library not installed{Colors.ENDC}")
        print("\nOr with uv:")
        print("  uv pip install mido")
        sys.exit(1)
    
    # Get file path from command line or prompt
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        print(f"\n{Colors.BOLD}MIDI File Analyzer{Colors.ENDC}")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} path/to/file.mid")
        print("\nOr drag and drop a MIDI file onto this script.")
        print("\nExample MIDI files can be found at:")
        print("  - https://bitmidi.com/")
        print("  - https://www.midiworld.com/")
        print("\nEnter path to MIDI file (or press Enter to exit): ", end="")
        
        filepath = input().strip()
        if not filepath:
            print("Exiting.")
            return
    
    # Analyze the file
    analyze_midi_file(filepath)


if __name__ == "__main__":
    main()

