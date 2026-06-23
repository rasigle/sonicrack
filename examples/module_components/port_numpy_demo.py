"""Example demonstrating Port class with numpy array support.

This example shows how the Port class can handle both scalar values
and numpy arrays for efficient batch audio processing.
"""

import numpy as np

from src.gui.core.port import Port


def main():
    print("=== Port Numpy Array Support Demo ===\n")

    # Example 1: Basic numpy array passing
    print("1. Basic Numpy Array Transfer:")
    output_port = Port("output", "Oscillator_Output")
    input_port = Port("input", "Effect_Input")

    input_port.connect(output_port)

    # Simulate audio buffer (512 samples at 440 Hz)
    sample_rate = 44100
    buffer_size = 512
    frequency = 440
    t = np.arange(buffer_size) / sample_rate
    audio_buffer = 0.5 * np.sin(2 * np.pi * frequency * t)

    output_port.write(audio_buffer)
    received = input_port.read()

    print(f"   Sent buffer shape: {audio_buffer.shape}")
    print(f"   Received buffer shape: {received.shape}")
    print(f"   Buffer identical: {np.allclose(audio_buffer, received)}\n")

    # Example 2: Mixing multiple audio buffers
    print("2. Mixing Multiple Audio Buffers:")
    mixer_input = Port("input", "Mixer_Input")
    osc1 = Port("output", "Oscillator_1_440Hz")
    osc2 = Port("output", "Oscillator_2_880Hz")
    osc3 = Port("output", "Oscillator_3_1320Hz")

    mixer_input.connect(osc1)
    mixer_input.connect(osc2)
    mixer_input.connect(osc3)

    # Generate three different frequencies
    buffer1 = 0.3 * np.sin(2 * np.pi * 440 * t)
    buffer2 = 0.2 * np.sin(2 * np.pi * 880 * t)
    buffer3 = 0.1 * np.sin(2 * np.pi * 1320 * t)

    osc1.write(buffer1)
    osc2.write(buffer2)
    osc3.write(buffer3)

    mixed = mixer_input.read()
    expected_mix = buffer1 + buffer2 + buffer3

    print(f"   Oscillator 1 amplitude: {np.max(np.abs(buffer1)):.3f}")
    print(f"   Oscillator 2 amplitude: {np.max(np.abs(buffer2)):.3f}")
    print(f"   Oscillator 3 amplitude: {np.max(np.abs(buffer3)):.3f}")
    print(f"   Mixed signal peak: {np.max(np.abs(mixed)):.3f}")
    print(f"   Mixing correct: {np.allclose(mixed, expected_mix)}\n")

    # Example 3: Mixing arrays with scalar gain
    print("3. Mixing Array with Scalar Gain:")
    mixer_input2 = Port("input", "Mixer_With_Gain")
    audio_source = Port("output", "Audio_Source")
    gain_control = Port("output", "Gain_Control")

    mixer_input2.connect(audio_source)
    mixer_input2.connect(gain_control)

    audio_signal = 0.8 * np.sin(2 * np.pi * 440 * t)
    gain = -0.3  # Reduce amplitude by 0.3

    audio_source.write(audio_signal)
    gain_control.write(gain)

    result = mixer_input2.read()

    print(f"   Original signal peak: {np.max(np.abs(audio_signal)):.3f}")
    print(f"   Gain adjustment: {gain:.3f}")
    print(f"   Result signal peak: {np.max(np.abs(result)):.3f}")
    print(f"   Expected peak: {np.max(np.abs(audio_signal + gain)):.3f}\n")

    # Example 4: Stereo buffer processing
    print("4. Stereo Buffer Processing:")
    stereo_input = Port("input", "Stereo_Input")
    left_channel = Port("output", "Left_Channel")
    right_channel = Port("output", "Right_Channel")

    stereo_input.connect(left_channel)
    stereo_input.connect(right_channel)

    # Create stereo buffers (2 channels, buffer_size samples)
    left_signal = np.array([0.5 * np.sin(2 * np.pi * 440 * t)])
    right_signal = np.array([0.5 * np.sin(2 * np.pi * 880 * t)])

    left_channel.write(left_signal)
    right_channel.write(right_signal)

    stereo_mix = stereo_input.read()

    print(f"   Left channel shape: {left_signal.shape}")
    print(f"   Right channel shape: {right_signal.shape}")
    print(f"   Mixed stereo shape: {stereo_mix.shape}")
    print(f"   Left peak: {np.max(np.abs(stereo_mix[0])):.3f}")
    print(f"   Right peak: {np.max(np.abs(stereo_mix[0])):.3f}\n")

    # Example 5: Performance comparison
    print("5. Performance with Large Buffers:")
    perf_input = Port("input", "Performance_Test")
    perf_outputs = [Port("output", f"Source_{i}") for i in range(8)]

    for port in perf_outputs:
        perf_input.connect(port)

    # Large buffer: 4096 samples, 8 sources
    large_buffer_size = 4096
    t_large = np.arange(large_buffer_size) / sample_rate

    import time

    start = time.perf_counter()

    for i, port in enumerate(perf_outputs):
        freq = 440 * (i + 1)
        buffer = 0.125 * np.sin(2 * np.pi * freq * t_large)
        port.write(buffer)

    mixed_result = perf_input.read()

    end = time.perf_counter()

    print(f"   Buffer size: {large_buffer_size} samples")
    print(f"   Number of sources: {len(perf_outputs)}")
    print(f"   Total samples processed: {large_buffer_size * len(perf_outputs)}")
    print(f"   Time taken: {(end - start) * 1000:.3f} ms")
    print(f"   Mixed signal shape: {mixed_result.shape}")
    print(f"   Mixed signal peak: {np.max(np.abs(mixed_result)):.3f}\n")

    # Example 6: Error handling
    print("6. Error Handling - Shape Mismatch:")
    error_input = Port("input", "Error_Test")
    port_a = Port("output", "Port_A")
    port_b = Port("output", "Port_B")

    error_input.connect(port_a)
    error_input.connect(port_b)

    port_a.write(np.array([1.0, 2.0, 3.0]))
    port_b.write(np.array([1.0, 2.0]))  # Different shape!

    try:
        error_input.read()
        print("   ✗ Should have raised ValueError!")
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}\n")

    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()
