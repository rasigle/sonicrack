# AudioPlayground Modular Synthesizer

A full-featured modular synthesizer with visual patching capabilities, inspired by analog modular synthesizers.

## Features

### 🎹 Modular Patching System
- **Visual Cable Patching**: Drag cables from output ports to input ports, just like a hardware modular synth
- **Flexible Routing**: Route any output to any compatible input
- **Real-time Connection**: Connect and disconnect modules on the fly
- **Module Library**: Easy access to all available modules

### 🎛️ Available Modules

1. **Oscillator**
   - Multiple waveforms: Sine, Square, Sawtooth, Triangle
   - Frequency control (20 Hz - 2000 Hz)
   - Amplitude control
   - Phase control

2. **ADSR Envelope**
   - Attack, Decay, Sustain, Release controls
   - Can modulate volume, pan, or other parameters

3. **Gain**
   - Simple gain/volume control (0.0 - 2.0)
   - No modulation input

4. **Pan**
   - Simple stereo positioning (-1.0 to 1.0)
   - No modulation input

5. **Clipper**
   - Hard clipping for distortion/limiting
   - Threshold control (0.1 - 2.0)

6. **Volume (Mod)**
   - Gain control (0.0 - 2.0)
   - Supports modulation input for envelope shaping

7. **Panner (Mod)**
   - Stereo positioning (-1.0 to 1.0)
   - Supports modulation for auto-panning effects

8. **Output**
   - Master volume control
   - Required for audio playback

### 📊 Visualizations
- **Real-time Waveform Display**: See your audio in the time domain
- **Spectrum Analyzer**: Frequency analysis with FFT
- **20 Hz update rate** for smooth visual feedback

### 🎵 Audio Engine
- **High-performance**: 200x+ real-time performance
- **Low-latency**: Real-time audio synthesis
- **44.1 kHz sample rate** (configurable)
- **Stereo output**

## Quick Start

### Installation

Make sure you have all dependencies installed:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install numpy scipy sounddevice pyqt6 pyqtgraph
```

### Running the Application

```bash
# From the AudioPlayground root directory
python examples/modular_synth_app.py
```

## How to Use

### Creating Your First Patch

1. **Add Modules**
   - Click buttons in the Module Library (left panel) to add modules to the canvas
   - Modules appear in the center canvas area

2. **Connect Modules**
   - Click and drag from an **output port** (red, on the right side of modules)
   - Release on an **input port** (green, on the left side of modules)
   - A cable appears connecting the two ports

3. **Configure Modules**
   - Use knobs (click and drag up/down to adjust)
   - Use sliders for linear controls
   - Use dropdowns to select options (e.g., waveform type)
   - **Patches compile automatically** when you change parameters or connections!

4. **Play Audio**
   - Click the "Play" button to start real-time synthesis
   - The patch compiles automatically if needed
   - Watch the waveform and spectrum displays update
   - Click "Stop" to stop playback

### Example Patches

#### Simple Sine Wave
1. Add: Oscillator → Output
2. Connect: Oscillator [Out] → Output [In]
3. Set Oscillator to "Sine" waveform
4. Play (auto-compiles)

#### Synth with Envelope
1. Add: Oscillator → Volume (Mod) → Output
2. Add: ADSR Envelope
3. Connect:
   - Oscillator [Out] → Volume (Mod) [In]
   - ADSR Envelope [Out] → Volume (Mod) [Mod]
   - Volume (Mod) [Out] → Output [In]
4. Adjust ADSR parameters (Attack: 0.1s, Decay: 0.2s, Sustain: 0.7, Release: 0.3s)
5. Play (auto-compiles)

#### Auto-Panned Synth
1. Add: Oscillator → Panner (Mod) → Output
2. Add: ADSR Envelope (for volume)
3. Add: Another Oscillator (for pan modulation)
4. Connect:
   - Oscillator 1 [Out] → Panner (Mod) [In]
   - Oscillator 2 [Out] → Panner (Mod) [Mod]
   - Panner (Mod) [Out] → Output [In]
5. Set Oscillator 2 to low frequency (e.g., 0.5 Hz) for slow panning
6. Play (auto-compiles)

#### Simple Gain Control
1. Add: Oscillator → Gain → Output
2. Connect:
   - Oscillator [Out] → Gain [In]
   - Gain [Out] → Output [In]
3. Adjust Gain knob to control volume (no envelope needed)
4. Play (auto-compiles)

#### Distortion Effect
1. Add: Oscillator → Clipper → Output
2. Connect:
   - Oscillator [Out] → Clipper [In]
   - Clipper [Out] → Output [In]
3. Set Oscillator amplitude to 0.8
4. Lower Clipper threshold (e.g., 0.3) for more distortion
5. Play (auto-compiles)

## Keyboard Shortcuts

- **Ctrl+N**: New patch (clears canvas)
- **Ctrl+Q**: Quit application
- **Delete/Backspace**: Delete selected cables

## Tips & Tricks

1. **Middle-click on knobs** to reset to default value
2. **Use mouse wheel** on knobs for fine adjustment
3. **Select cables** by clicking on them (they turn yellow)
4. **Multiple oscillators** can be connected to create complex timbres
5. **Experiment** with different modulation sources

## Troubleshooting

### No Sound
- Ensure you have an Output module in your patch
- Check that modules are properly connected
- Verify master volume is not at zero
- The patch compiles automatically when you connect modules or change parameters

### Audio Glitches
- Check CPU usage (patch might be too complex)
- Ensure your audio drivers are up to date
- Try increasing buffer size (in code: `AudioEngine(buffer_size=4096)`)

### Visual Performance
- Reduce visualization update rate if needed
- Close other applications to free up resources

## Architecture

The modular synth consists of several key components:

- **PatchCanvas**: Visual patching interface with drag-and-drop cables
- **ModuleWidget**: Base class for all modules
- **Port**: Connection points on modules
- **Cable**: Visual representation of signal routing
- **PatchCompiler**: Converts visual patch to audio signal chain
- **AudioEngine**: Real-time audio generation and playback
- **Visualizations**: Waveform and spectrum analysis

## Future Enhancements

- [ ] MIDI input support
- [ ] Preset save/load system
- [ ] More module types (filters, effects, etc.)
- [ ] LFO with multiple waveforms
- [ ] Sequencer/arpeggiator
- [ ] Wavetable oscillators
- [ ] Multi-voice polyphony
- [ ] Recording to WAV file
- [ ] Module browser with search
- [ ] Zoom and pan canvas
- [ ] Undo/redo functionality

## Credits

Built on the AudioPlayground synthesis engine - a high-performance audio synthesis framework with world-class vectorized performance.

---

**Enjoy creating sounds!** 🎵🎶

