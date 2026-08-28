"""Shared In→Out shell for sample-rate-aware effect wrappers."""

from __future__ import annotations

from collections.abc import Sequence

from sonicrack.gui.modules.effects._cv_modulation import (
    ControlRateCvSpec,
    apply_control_rate_cv,
)
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.runtime.helpers import read_samples
from sonicrack.runtime.specs import RuntimeParameters


class SimpleEffectModule(ModuleWidget):
    """In→Out effect that applies control-rate parameters once per block."""

    def _setup_effect_io(self, component: object) -> None:
        """Add standard audio I/O and attach the DSP component."""
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.component = component

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def control_rate_specs(self) -> Sequence[ControlRateCvSpec]:
        """Return control-rate parameter specs applied each audio block."""
        return ()

    def apply_runtime_parameters(
        self, parameters: RuntimeParameters, num_samples: int
    ) -> None:
        """Push knob/CV values onto the DSP component for this block."""
        apply_control_rate_cv(
            self.component, parameters, self.control_rate_specs(), num_samples
        )

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if self._require_input_or_silence(num_samples):
            return
        self.apply_runtime_parameters(parameters, num_samples)
        self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
