# MIDI Input Demo

This example demonstrates how to receive and display MIDI messages in real-time from a connected MIDI controller or keyboard.

## Requirements

Install the required MIDI libraries:

```bash
pip install mido python-rtmidi
```

Or with uv:

```bash
uv pip install mido python-rtmidi
```

## Usage

1. **Connect a MIDI device** (keyboard, controller, etc.) to your computer

2. **Run the example**:
   ```bash
   python examples/midi_input_demo.py
   ```

3. **Select your device** from the list (or press Enter to use the first one)

4. **Play notes or move controls** on your MIDI device to see messages appear!

5. **Press Ctrl+C** to exit

## What You'll See

The demo displays:

### Note Messages
```
[  0.52s] Ch 1 NOTE ON  C4   (#60, 261.63Hz) Velocity: 100 (79%)
[  0.75s] Ch 1 NOTE OFF C4   (#60)
```

### Control Change (CC)
```
[  1.23s] Ch 1 CC       Mod Wheel            =  64 (50%)
[  2.45s] Ch 1 CC       Sustain Pedal        = ON
```

### Pitch Bend
```
[  3.12s] Ch 1 PITCHBND ↑  +2048 (+0.50 semitones)
```

### Program Change
```
[  4.00s] Ch 1 PROGRAM  → 5
```

## Features

- ✅ **Color-coded output** - Different colors for different message types
- ✅ **Note names** - Shows "C4" instead of just "60"
- ✅ **Frequencies** - Displays Hz for each note
- ✅ **Normalized values** - Shows percentages (0-100%)
- ✅ **Common CC names** - "Mod Wheel" instead of "CC 1"
- ✅ **Sustain pedal detection** - Shows ON/OFF state
- ✅ **Real-time timestamps** - Shows when each message arrived

## Message Types

The demo handles all common MIDI message types:

| Type | Description | Example |
|------|-------------|---------|
| **Note On** | Key pressed | Middle C at 80% velocity |
| **Note Off** | Key released | Middle C released |
| **Control Change** | Knob/slider moved | Mod wheel to 50% |
| **Pitch Bend** | Pitch wheel | Bend up 1 semitone |
| **Program Change** | Preset selected | Switch to program 5 |
| **Aftertouch** | Key pressure | Apply 60% pressure |

## Common CC Controllers

| CC# | Name | Description |
|-----|------|-------------|
| 1 | Mod Wheel | Modulation depth |
| 7 | Volume | Channel volume |
| 10 | Pan | Left/right position |
| 11 | Expression | Expression pedal |
| 64 | Sustain | Sustain pedal (on/off) |
| 74 | Cutoff | Filter cutoff (common) |

## Troubleshooting

### No devices found
- Make sure your MIDI device is connected
- Check that drivers are installed (especially on Windows)
- Try unplugging and reconnecting the device

### Permission denied (Linux)
Add your user to the audio group:
```bash
sudo usermod -a -G audio $USER
```
Then log out and back in.

### Device not appearing
- Close other apps using MIDI (DAWs, etc.)
- Restart your computer
- Update device drivers

## Code Structure

The example demonstrates:

1. **Device enumeration** - `MIDIInput.list_devices()`
2. **Opening a device** - `MIDIInput(device_name)`
3. **Callback handling** - `midi.start(on_message)`
4. **Message conversion** - Automatic translation to our message types
5. **Clean shutdown** - Using context manager (`with` statement)

## Next Steps

Once you understand MIDI input, you can:

- **Trigger oscillators** from MIDI notes
- **Control parameters** with CC messages
- **Build a polyphonic synth** with voice management
- **Record MIDI** to files
- **Create a MIDI-controlled GUI module**

See the MIDI roadmap in `documentation/MIDI_ROADMAP.md` for the complete plan.

## Example Output

```
=== MIDI Input Demo ===

Found 2 MIDI input device(s):

  1. USB MIDI Keyboard
  2. Virtual MIDI Bus 1

Using: USB MIDI Keyboard

Listening for MIDI messages... (Press Ctrl+C to exit)

================================================================================
[  0.00s] Ch 1 NOTE ON  C4   (#60, 261.63Hz) Velocity: 100 (79%)
[  0.15s] Ch 1 NOTE ON  E4   (#64, 329.63Hz) Velocity:  95 (75%)
[  0.28s] Ch 1 NOTE ON  G4   (#67, 392.00Hz) Velocity:  98 (77%)
[  1.52s] Ch 1 NOTE OFF C4   (#60)
[  1.54s] Ch 1 NOTE OFF E4   (#64)
[  1.55s] Ch 1 NOTE OFF G4   (#67)
[  2.10s] Ch 1 CC       Mod Wheel            =  64 (50%)
[  2.85s] Ch 1 CC       Sustain Pedal        = ON
[  3.45s] Ch 1 PITCHBND ↑  +1024 (+0.25 semitones)
[  4.12s] Ch 1 PITCHBND •      0 (+0.00 semitones)

Stopped.
```

Enjoy exploring MIDI! 🎹

