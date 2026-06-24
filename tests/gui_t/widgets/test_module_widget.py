from __future__ import annotations

from typing import Any

import numpy as np

from src.engine.audio_component import AudioComponent
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.widgets.module_widget import ModuleWidget


class _SourceWidget(ModuleWidget):
    metadata = ModuleMetadata("Test Source", ModuleCategory.SOURCE)

    def __init__(self):
        super().__init__()
        self.out_port = self.add_output("Out")
        self.controls_widget = self._create_controls_container()
        self._create_portwidgets()

    def create_engine_component(
        self,
        input_components: list[AudioComponent] | None = None,
        modulation_components: dict[str, AudioComponent] | None = None,
    ) -> AudioComponent | None:
        del input_components, modulation_components
        return None

    def process(self, num_samples: int = 1):
        self.out_port.write(np.ones(num_samples, dtype=np.float32))


class _ModifierWidget(ModuleWidget):
    metadata = ModuleMetadata("Test Modifier", ModuleCategory.MODIFIER)

    def __init__(self):
        super().__init__()
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.controls_widget = self._create_controls_container()
        self._create_portwidgets()

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def create_engine_component(
        self,
        input_components: list[AudioComponent] | None = None,
        modulation_components: dict[str, AudioComponent] | None = None,
    ) -> AudioComponent | None:
        del input_components, modulation_components
        return None

    def process(self, num_samples: int = 1):
        value = self.input_ports[0].port.read(num_samples)
        self.out_port.write(np.asarray(value) * 2)


def test_inactive_source_outputs_silence(qapp: Any):
    del qapp
    module = _SourceWidget()

    module.ensure_samples_ready(4)
    assert np.allclose(module.out_port.value, np.ones(4, dtype=np.float32))

    module.set_active(False)
    module.ensure_samples_ready(4)

    assert not module.controls_widget.isEnabled()
    assert np.allclose(module.out_port.value, np.zeros(4, dtype=np.float32))


def test_inactive_modifier_passes_required_input_through(qapp: Any):
    del qapp
    source = _SourceWidget()
    modifier = _ModifierWidget()
    source.output_ports[0].connect(modifier.input_ports[0])

    modifier.ensure_samples_ready(4)
    assert np.allclose(modifier.out_port.value, np.full(4, 2.0, dtype=np.float32))

    modifier.set_active(False)
    source.invalidate_cache()
    modifier.invalidate_cache()
    modifier.ensure_samples_ready(4)

    assert np.allclose(modifier.out_port.value, np.ones(4, dtype=np.float32))
