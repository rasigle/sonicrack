import logging
from typing import Any

import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab import Volume
from soniclab.dsp.modifiers import ModulatedVolume

from sonicrack.gui.core.module import ModuleCategory, ModuleMetadata
from sonicrack.gui.core.runtime import RuntimeParameters
from sonicrack.gui.core.runtime_helpers import float_parameter, read_samples, silence
from sonicrack.gui.module_registry import register_module
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget

logger = logging.getLogger(__name__)


@register_module()
class VCAModule(ModuleWidget):
    """Voltage-Controlled Amplifier (VCA) with amplitude modulation.

    A VCA controls the amplitude/volume of an input signal. It can be controlled
    manually with the amplitude knob, or via a control voltage (CV) input.

    Gain controls the final output level. When CV is connected, the CV Attn knob
    controls how strongly the CV signal influences the VCA before that final gain.
    At 0.0, CV has no influence. At 1.0, CV has maximum influence.

    Inputs:
        - In: Audio input signal
        - CV: Control voltage input (0.0 to 1.0) for amplitude modulation

    Outputs:
        - Out: Amplitude-controlled audio output

    Parameters:
        - Amplitude: Final output gain (0.0 to 1.0)
        - CV Attn: CV influence amount (0.0 to 1.0)
    """

    runtime_kind = "vca"

    metadata = ModuleMetadata(
        title="VCA",
        category=ModuleCategory.MODIFIER,
        description="Voltage-Controlled Amplifier with CV modulation",
        version="1.0.0",
        author="SonicRack",
    )

    def __init__(self):
        """Initialize VCA module."""
        super().__init__(
            width=220,
            height=180,
            color=QColor(100, 140, 180),
        )

        # Add ports
        self.in_port = self.add_input("In")
        self.cv_port = self.add_input("CV In")
        self.out_port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Amplitude control
        knobs_layout = QHBoxLayout()

        self.cv_attn_knob = Knob(
            label="CV Attn",
            min_value=0.0,
            max_value=1.0,
            default_value=1.0,
            logarithmic=False,
        )
        self.cv_attn_knob.setToolTip("CV influence amount: 0 = Gain only, 1 = CV only")
        self.cv_attn_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "cv_attenuation", self.cv_attn_knob.get_value()
            )
        )
        knobs_layout.addWidget(self.cv_attn_knob)

        self.gain_knob = Knob(
            label="Gain",
            min_value=0.0,
            max_value=1.0,
            default_value=1.0,
            logarithmic=False,
        )
        self.gain_knob.setToolTip("Final VCA output gain (0.0 to 1.0)")
        self.gain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("amplitude", self.gain_knob.get_value())
        )
        knobs_layout.addWidget(self.gain_knob)
        layout.addLayout(knobs_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("amplitude", self.gain_knob)
        self.register_parameter("cv_attenuation", self.cv_attn_knob)

        self.component = self.create_engine_component()

    def get_required_inputs(self) -> list[str]:
        """VCA requires audio input on 'In' port."""
        return ["In"]

    def get_modulation_inputs(self) -> list[str]:
        """VCA accepts modulation on CV port."""
        return ["CV In"]

    def get_cv_range(self, port_name: str = "CV In") -> tuple[float, float]:
        """VCA CV port expects unipolar range [0, 1] for amplitude modulation.

        Returns:
            (0.0, 1.0) - unipolar range (0 = silence, 1 = full amplitude)
        """
        return 0.0, 1.0

    def update_knob_state(self):
        """Update control tooltips based on CV port connection."""
        # Check CV port connection
        cv_port = None
        for port in self.input_ports:
            if port.port_name == "CV In":
                cv_port = port
                break

        has_cv = cv_port and len(cv_port.cables) > 0

        logger.info(
            f"VCA update_knob_state: CV port has "
            f"{len(cv_port.cables) if cv_port else 0} cables, has_cv={has_cv}"
        )

        self.gain_knob.setEnabled(True)
        self.gain_knob.setStyleSheet("")
        if has_cv:
            self.gain_knob.setToolTip("Final VCA output gain after CV modulation")
            self.cv_attn_knob.setToolTip(
                "CV influence amount: 0 = Gain only, 1 = CV only"
            )
            logger.info("VCA: CV connected; Gain remains editable as output gain")
        else:
            self.gain_knob.setToolTip("Final VCA output gain (0.0 to 1.0)")
            self.cv_attn_knob.setToolTip("CV influence when CV input is connected")
            logger.info("VCA: No CV connected; Gain controls amplitude")

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the VCA component.

        If CV is connected, creates a ModulatedVolume that responds to CV input.
        Otherwise, creates a simple Volume with manual amplitude control.

        Args:
            input_components: Audio input signal (required)
            modulation_components: CV modulation input (optional)

        Returns:
            Volume or ModulatedVolume component (patch compiler wraps in Chain)
        """
        amplitude = self.gain_knob.get_value()
        cv_attenuation = self.cv_attn_knob.get_value()

        # Check for CV modulation
        cv_modulator = (
            modulation_components.get("CV In")
            if modulation_components and isinstance(modulation_components, dict)
            else None
        )
        has_cv = cv_modulator is not None

        logger.info(
            f"VCA: Creating component with amplitude={amplitude:.3f}, "
            f"cv_attenuation={cv_attenuation:.3f}, has_cv={has_cv}, "
            f"modulation_components={modulation_components}"
        )

        # Update UI state
        if has_cv:
            self.gain_knob.setEnabled(True)
            self.gain_knob.setStyleSheet("")
            self.gain_knob.setToolTip("Final VCA output gain after CV modulation")
            self.cv_attn_knob.setToolTip(
                "CV influence amount: 0 = Gain only, 1 = CV only"
            )
            logger.info("VCA create_engine_component: CV connected")

            # Create ModulatedVolume with CV influence control.
            # Patch compiler will wrap: Chain(input_signal, ModulatedVolume)
            if cv_attenuation <= 0.0:
                volume_component = Volume(amplitude=amplitude)
                logger.info(
                    f"VCA: Returning manual Volume component: {volume_component}, "
                    f"type={type(volume_component).__name__}"
                )
                return volume_component

            influenced_cv = _CVInfluenceAmplitude(
                cv_modulator,
                output_gain=amplitude,
                influence=cv_attenuation,
            )
            modulated_component = ModulatedVolume(
                influenced_cv, modulation_target="amplitude"
            )
            logger.info(
                f"VCA: Returning ModulatedVolume component: {modulated_component}, "
                f"type={type(modulated_component).__name__}"
            )
            return modulated_component

        # Static volume component (no CV)
        self.gain_knob.setEnabled(True)
        self.gain_knob.setStyleSheet("")
        self.gain_knob.setToolTip("Final VCA output gain (0.0 to 1.0)")
        self.cv_attn_knob.setToolTip("CV influence when CV input is connected")
        logger.info("VCA create_engine_component: Amplitude knob ENABLED (no CV)")

        # Create simple Volume with manual control
        # Patch compiler will wrap: Chain(input_signal, Volume)
        volume_component = Volume(amplitude=amplitude)
        logger.info(
            f"VCA: Returning Volume component: {volume_component}, "
            f"type={type(volume_component).__name__}"
        )
        return volume_component

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Apply manual or CV-controlled amplitude for one render cycle."""
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        input_signal = read_samples(self.in_port, num_samples)
        manual_amplitude = float_parameter(
            parameters, "amplitude", self.gain_knob.get_value
        )
        if self.cv_port.is_connected:
            cv_attenuation = float_parameter(
                parameters, "cv_attenuation", self.cv_attn_knob.get_value
            )
            cv_amplitude = read_samples(self.cv_port, num_samples)
            amplitude = _apply_cv_influence(
                cv_amplitude,
                cv_attenuation,
                output_gain=manual_amplitude,
            )
        else:
            amplitude = manual_amplitude
        self.out_port.write(input_signal * amplitude)


def _apply_cv_influence(
    cv_amplitude: float | np.ndarray,
    influence: float,
    *,
    output_gain: float,
) -> float | np.ndarray:
    """Apply CV influence, then final output gain."""
    amount = max(0.0, min(1.0, float(influence)))
    modulation = (1.0 - amount) + (cv_amplitude * amount)
    return np.clip(modulation * float(output_gain), 0.0, 1.0)


class _CVInfluenceAmplitude:
    """Iterator/generator adapter that blends manual gain with incoming CV."""

    def __init__(self, source: Any, *, output_gain: float, influence: float):
        self._source = source
        self._iterator = iter(source)
        self._output_gain = float(output_gain)
        self._influence = float(influence)

    def __iter__(self):
        self._iterator = iter(self._source)
        return self

    def __next__(self):
        return float(
            _apply_cv_influence(
                next(self._iterator),
                self._influence,
                output_gain=self._output_gain,
            )
        )

    def get_samples(self, num_samples: int, **kwargs) -> np.ndarray:
        if hasattr(self._source, "get_samples_vectorized") and not kwargs:
            cv_values = self._source.get_samples_vectorized(num_samples)
        elif hasattr(self._source, "get_samples"):
            cv_values = self._source.get_samples(num_samples, **kwargs)
        else:
            cv_values = np.array(
                [next(self._iterator) for _ in range(num_samples)],
                dtype=np.float32,
            )
        return np.asarray(
            _apply_cv_influence(
                cv_values,
                self._influence,
                output_gain=self._output_gain,
            ),
            dtype=np.float32,
        )

    def trigger_release(self):
        if hasattr(self._source, "trigger_release"):
            self._source.trigger_release()

    @property
    def ended(self):
        return bool(getattr(self._source, "ended", False))
