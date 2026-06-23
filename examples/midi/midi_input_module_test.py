"""Test the MIDI Input module in the modular synthesizer GUI.

This script demonstrates the MIDI Input module by creating a simple patch:
MIDI Input (Freq) → Oscillator
MIDI Input (Gate) → ADSR Envelope → Volume (Mod)
→ Output

This allows you to play the synthesizer with a MIDI keyboard!
"""

import sys

from PyQt6.QtWidgets import QApplication

# Import the modular synth window
from src.gui.main_window import ModularSynthWindow


def main():
    """Run the modular synth with instructions for MIDI testing."""
    app = QApplication(sys.argv)

    window = ModularSynthWindow()
    window.show()

    # Print instructions
    print("\n" + "=" * 70)
    print("MIDI Input Module Test")
    print("=" * 70)
    print("\nTo test the MIDI Input module:")
    print("1. Add a 'MIDI Input' module from the module library")
    print("2. Select your MIDI device from the dropdown")
    print("3. Click 'Start' to begin receiving MIDI")
    print("4. Add an 'Oscillator' module")
    print("5. Connect MIDI Input 'Freq' → Oscillator 'Freq'")
    print("6. Add an 'ADSR Envelope' module")
    print("7. Connect MIDI Input 'Gate' → ADSR 'Gate'")
    print("8. Add a 'Volume (Mod)' module")
    print("9. Connect Oscillator 'Out' → Volume 'In'")
    print("10. Connect ADSR 'Out' → Volume 'Mod'")
    print("11. Add an 'Output' module")
    print("12. Connect Volume 'Out' → Output 'In'")
    print("13. Click 'Compile Patch'")
    print("14. Click 'Play'")
    print("15. Play notes on your MIDI keyboard!")
    print("\nYou should see:")
    print("- Current note displayed on MIDI Input module")
    print("- Sound output when you press keys")
    print("- Envelope controlling the volume")
    print("=" * 70 + "\n")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
