from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import QPointF

from src.engine.core.component import AudioComponent
from src.gui.audio_engine import AudioEngine
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters, get_runtime_spec
from src.gui.core.runtime_helpers import read_samples
from src.gui.modules.visualization.waveform import WaveformModule
from src.gui.patch_canvas import PatchCanvas
from src.gui.widgets.module_widget import ModuleWidget


class _OnesComponent:
    def __init__(self, owner: _SourceWidget):
        self.owner = owner

    def get_samples(self, num_samples: int) -> np.ndarray:
        self.owner.process_count += 1
        return np.ones(num_samples, dtype=np.float32)


class _DoubleComponent:
    def __call__(self, samples: np.ndarray) -> np.ndarray:
        return samples * 2


class _ValueControl:
    def __init__(self, value: float):
        self.value = value

    def get_value(self) -> float:
        return self.value


class _SourceWidget(ModuleWidget):
    metadata = ModuleMetadata("Test Source", ModuleCategory.SOURCE)
    runtime_kind = "single_source"

    def __init__(self):
        super().__init__()
        self.out_port = self.add_output("Out")
        self.controls_widget = self._create_controls_container()
        self._create_portwidgets()
        self.component = None
        self.process_count = 0
        self.invalidate_count = 0

    def create_engine_component(
        self,
        input_components: list[AudioComponent] | None = None,
        modulation_components: dict[str, AudioComponent] | None = None,
    ) -> Any:
        del input_components, modulation_components
        return _OnesComponent(self)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        del parameters
        if self.component is None:
            self.component = self.create_engine_component()
        self.out_port.write(self.component.get_samples(num_samples))

    def invalidate_cache(self):
        self.invalidate_count += 1
        super().invalidate_cache()


class _VisualizerWidget(ModuleWidget):
    metadata = ModuleMetadata("Test Visualizer", ModuleCategory.VISUALIZATION)
    is_processing_module = False

    def __init__(self):
        super().__init__()
        self.in_port = self.add_input("In")
        self.controls_widget = self._create_controls_container()
        self._create_portwidgets()

    def create_engine_component(
        self,
        input_components: list[AudioComponent] | None = None,
        modulation_components: dict[str, AudioComponent] | None = None,
    ) -> Any:
        del input_components, modulation_components
        return None


class _ModifierWidget(ModuleWidget):
    metadata = ModuleMetadata(
        "Test Modifier",
        ModuleCategory.MODIFIER,
        description="Test modifier description",
    )
    runtime_kind = "component_modifier"

    def __init__(self):
        super().__init__()
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.controls_widget = self._create_controls_container()
        self._create_portwidgets()
        self.component = None
        self.register_parameter("gain_db", _ValueControl(-3.0))

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def get_parameters(self) -> dict[str, Any]:
        return {"gain_db": -3.0, "ignored": 99}

    def create_engine_component(
        self,
        input_components: list[AudioComponent] | None = None,
        modulation_components: dict[str, AudioComponent] | None = None,
    ) -> Any:
        del input_components, modulation_components
        return _DoubleComponent()

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        del parameters
        if not self.in_port.is_connected:
            self.out_port.write(np.zeros(num_samples, dtype=np.float32))
            return
        if self.component is None:
            self.component = self.create_engine_component()
        self.out_port.write(self.component(read_samples(self.in_port, num_samples)))


def test_inactive_source_outputs_silence(qapp: Any):
    del qapp
    engine = AudioEngine()
    source = _SourceWidget()
    visualizer = _VisualizerWidget()
    source.output_ports[0].connect(visualizer.input_ports[0])
    engine.add_module(source)
    engine.add_module(visualizer)

    engine.render_ports([visualizer.in_port], 4)
    assert np.allclose(source.out_port.value, np.ones(4, dtype=np.float32))

    source.set_active(False)
    engine.render_ports([visualizer.in_port], 4)

    assert not source.controls_widget.isEnabled()
    assert np.allclose(source.out_port.value, np.zeros(4, dtype=np.float32))


def test_port_widgets_show_direction_tooltips(qapp: Any):
    del qapp
    modifier = _ModifierWidget()

    assert modifier.input_ports[0].toolTip() == "Test Modifier Input: In"
    assert modifier.output_ports[0].toolTip() == "Test Modifier Output: Out"


def test_module_header_tooltip_uses_module_description(qapp: Any):
    del qapp
    modifier = _ModifierWidget()

    assert modifier.toolTip() == ""
    modifier._update_tooltip_at(QPointF(40, 10))

    assert modifier.toolTip() == "Test modifier description"


def test_module_power_button_tooltip_is_scoped_to_button(qapp: Any):
    del qapp
    modifier = _ModifierWidget()

    modifier._update_tooltip_at(modifier._power_button_rect().center())
    assert modifier.toolTip() == "Click the power button to bypass this module."

    modifier._update_tooltip_at(QPointF(40, modifier._title_bar_height() + 10))
    assert modifier.toolTip() == ""


def test_patch_canvas_enables_mouse_tracking_for_graphics_tooltips(qapp: Any):
    del qapp
    canvas = PatchCanvas()

    assert canvas.hasMouseTracking()
    assert canvas.viewport().hasMouseTracking()


def test_inactive_modifier_passes_required_input_through(qapp: Any):
    del qapp
    engine = AudioEngine()
    source = _SourceWidget()
    modifier = _ModifierWidget()
    source.output_ports[0].connect(modifier.input_ports[0])
    engine.add_module(source)
    engine.add_module(modifier)

    engine.render_ports([modifier.out_port], 4)
    assert np.allclose(modifier.out_port.value, np.full(4, 2.0, dtype=np.float32))

    modifier.set_active(False)
    engine.render_ports([modifier.out_port], 4)

    assert np.allclose(modifier.out_port.value, np.ones(4, dtype=np.float32))


def test_monitor_runtime_renders_visualizer_only_sink(qapp: Any):
    del qapp
    engine = AudioEngine()
    source = _SourceWidget()
    visualizer = _VisualizerWidget()
    source.output_ports[0].connect(visualizer.input_ports[0])
    engine.add_module(source)
    engine.add_module(visualizer)

    engine.render_monitor_sinks()

    assert source.process_count == 1
    assert source.invalidate_count == 1
    assert np.allclose(source.out_port.peek_recent(4), np.ones(4, dtype=np.float32))


def test_render_context_renders_shared_source_once_for_parallel_sinks(qapp: Any):
    del qapp
    engine = AudioEngine()
    source = _SourceWidget()
    visualizer_a = _VisualizerWidget()
    visualizer_b = _VisualizerWidget()
    source.output_ports[0].connect(visualizer_a.input_ports[0])
    source.output_ports[0].connect(visualizer_b.input_ports[0])
    engine.add_module(source)
    engine.add_module(visualizer_a)
    engine.add_module(visualizer_b)

    values = engine.render_ports(
        [visualizer_a.in_port, visualizer_b.in_port],
        4,
    )

    assert source.process_count == 1
    assert len(values) == 2
    assert np.allclose(values[0], np.ones(4, dtype=np.float32))
    assert np.allclose(values[1], np.ones(4, dtype=np.float32))


def test_render_context_mixes_parallel_sources(qapp: Any):
    del qapp
    engine = AudioEngine()
    source_a = _SourceWidget()
    source_b = _SourceWidget()
    visualizer = _VisualizerWidget()
    source_a.output_ports[0].connect(visualizer.input_ports[0])
    source_b.output_ports[0].connect(visualizer.input_ports[0])
    engine.add_module(source_a)
    engine.add_module(source_b)
    engine.add_module(visualizer)

    values = engine.render_ports([visualizer.in_port], 4)

    assert source_a.process_count == 1
    assert source_b.process_count == 1
    assert np.allclose(values[0], np.full(4, 2.0, dtype=np.float32))


def test_render_context_returns_cached_port_read_copies(qapp: Any):
    del qapp
    engine = AudioEngine()
    source = _SourceWidget()
    visualizer = _VisualizerWidget()
    source.output_ports[0].connect(visualizer.input_ports[0])
    engine.add_module(source)
    engine.add_module(visualizer)

    plan = engine.compile_render_plan([visualizer.in_port])
    with engine.render_context(4) as context:
        context.render_plan(plan)
        first = visualizer.in_port.read(4)
        second = visualizer.in_port.read(4)

    assert source.process_count == 1
    assert isinstance(first, np.ndarray)
    assert isinstance(second, np.ndarray)
    first[0] = 99.0
    assert second[0] == 1.0


def test_render_plan_renders_dependency_before_modifier(qapp: Any):
    del qapp
    engine = AudioEngine()
    source = _SourceWidget()
    modifier = _ModifierWidget()
    visualizer = _VisualizerWidget()
    source.output_ports[0].connect(modifier.input_ports[0])
    modifier.output_ports[0].connect(visualizer.input_ports[0])
    engine.add_module(source)
    engine.add_module(modifier)
    engine.add_module(visualizer)

    values = engine.render_ports([visualizer.in_port], 4)

    assert source.process_count == 1
    assert np.allclose(modifier.out_port.value, np.full(4, 2.0, dtype=np.float32))
    assert np.allclose(values[0], np.full(4, 2.0, dtype=np.float32))


def test_render_plan_stores_runtime_specs(qapp: Any):
    del qapp
    engine = AudioEngine()
    source = _SourceWidget()
    visualizer = _VisualizerWidget()
    source.output_ports[0].connect(visualizer.input_ports[0])

    plan = engine.compile_render_plan([visualizer.in_port])

    assert len(plan.nodes) == 1
    assert plan.nodes[0].module is source
    assert plan.nodes[0].spec.kind == "single_source"
    assert plan.nodes[0].spec.processor is not None
    assert plan.nodes[0].input_ports == ()
    assert plan.nodes[0].output_ports == (source.out_port,)


def test_render_plan_binds_declared_node_ports(qapp: Any):
    del qapp
    engine = AudioEngine()
    source = _SourceWidget()
    modifier = _ModifierWidget()
    visualizer = _VisualizerWidget()
    modifier.runtime_kind = "volume"
    source.output_ports[0].connect(modifier.input_ports[0])
    modifier.output_ports[0].connect(visualizer.input_ports[0])

    plan = engine.compile_render_plan([visualizer.in_port])
    modifier_node = next(node for node in plan.nodes if node.module is modifier)

    assert modifier_node.spec.input_names == ("In",)
    assert modifier_node.input_ports == (modifier.in_port,)
    assert modifier_node.output_ports == (modifier.out_port,)
    assert modifier_node.dependencies == (source,)


def test_port_read_no_longer_triggers_upstream_processing(qapp: Any):
    del qapp
    source = _SourceWidget()
    visualizer = _VisualizerWidget()
    source.output_ports[0].connect(visualizer.input_ports[0])

    value = visualizer.in_port.read(4)

    assert source.process_count == 0
    assert value == 0.0


def test_waveform_update_is_passive_and_does_not_render_upstream(qapp: Any):
    del qapp
    source = _SourceWidget()
    waveform = WaveformModule()
    source.output_ports[0].connect(waveform.input_ports[0])
    source.out_port.write(np.linspace(-1.0, 1.0, 16, dtype=np.float32))

    waveform._update_display()

    assert source.process_count == 0
    assert waveform.waveform_display.samples is not None
    assert np.allclose(
        waveform.waveform_display.samples,
        np.linspace(-1.0, 1.0, 16, dtype=np.float32),
    )


def test_runtime_spec_uses_explicit_kind_not_display_title(qapp: Any):
    del qapp
    source = _SourceWidget()
    source.metadata = ModuleMetadata("Renamed Source", ModuleCategory.SOURCE)

    assert get_runtime_spec(source).kind == "single_source"


def test_runtime_spec_declares_processor_and_ports(qapp: Any):
    del qapp
    modifier = _ModifierWidget()

    spec = get_runtime_spec(modifier)

    assert spec.kind == "component_modifier"
    assert spec.processor is not None
    assert spec.input_names == ("In",)
    assert spec.output_names == ("Out",)
    assert spec.parameter_names == ("gain_db",)


def test_render_plan_stores_runtime_parameter_snapshot(qapp: Any):
    del qapp
    engine = AudioEngine()
    source = _SourceWidget()
    modifier = _ModifierWidget()
    visualizer = _VisualizerWidget()
    modifier.runtime_kind = "volume"
    source.output_ports[0].connect(modifier.input_ports[0])
    modifier.output_ports[0].connect(visualizer.input_ports[0])

    plan = engine.compile_render_plan([visualizer.in_port])
    modifier_node = next(node for node in plan.nodes if node.module is modifier)

    assert modifier_node.parameters == {"gain_db": -3.0}
