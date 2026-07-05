"""Regression tests for AudioEngine render-plan caching."""

from __future__ import annotations

import numpy as np

from src.gui.audio_engine import AudioEngine
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.port import Port
from src.gui.core.runtime import RuntimeModuleSpec


class _PortWidgetStub:
    def __init__(self, port: Port) -> None:
        self.port = port


class _SourceModule:
    runtime_kind = "test_source"
    is_active = True
    is_processing_module = True
    metadata = ModuleMetadata(
        title="Test Source",
        category=ModuleCategory.SOURCE,
        description="Test source",
    )

    def __init__(self) -> None:
        self.out = Port("output", "Out", parent_module=self)
        self.inputs = {}
        self.outputs = {"Out": self.out}
        self.input_ports = []
        self.output_ports = [_PortWidgetStub(self.out)]
        self.render_count = 0
        self.gain = 1.0

    def get_runtime_spec(self) -> RuntimeModuleSpec:
        return RuntimeModuleSpec(
            kind=self.runtime_kind,
            processor=self.process_runtime,
            output_names=("Out",),
            parameter_names=("gain",),
        )

    def process_runtime(self, num_samples: int, parameters: object) -> None:
        self.render_count += 1
        gain = float(parameters["gain"])
        self.out.write(np.full(num_samples, self.render_count * gain, dtype=np.float32))

    def get_parameters(self) -> dict[str, object]:
        return {"gain": self.gain}

    def get_required_inputs(self) -> list[str]:
        return []

    def _find_port_by_name(self, _port_name: str) -> None:
        return None

    def invalidate_cache(self) -> None:
        return


def test_audio_engine_reuses_and_invalidates_render_plan(qapp):
    del qapp
    engine = AudioEngine()
    source = _SourceModule()
    sink = Port("input", "In")
    source.out.connect(sink)
    engine.add_module(source)

    compile_count = 0
    original_compile = engine.compile_render_plan

    def counting_compile(ports):
        nonlocal compile_count
        compile_count += 1
        return original_compile(ports)

    engine.compile_render_plan = counting_compile

    first = engine.render_ports([sink], 4)[0]
    second = engine.render_ports([sink], 4)[0]
    source.gain = 10.0
    third = engine.render_ports([sink], 4)[0]

    assert compile_count == 1
    np.testing.assert_allclose(first, [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(second, [2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(third, [30.0, 30.0, 30.0, 30.0])

    engine.mark_graph_changed()
    fourth = engine.render_ports([sink], 4)[0]

    assert compile_count == 2
    np.testing.assert_allclose(fourth, [40.0, 40.0, 40.0, 40.0])
