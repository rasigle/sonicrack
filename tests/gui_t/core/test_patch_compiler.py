"""Tests for PatchCompiler validation and tree behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from src.gui.core.module import AudioModule, ModuleCategory
from src.gui.core.patch_compiler import PatchCompiler
from src.gui.widgets.port_widget import PortWidget


@dataclass
class _Metadata:
    title: str
    category: ModuleCategory


class _Port:
    def __init__(self, parent_module, port_name: str, port_type: str):
        self.parent_module = parent_module
        self.port_name = port_name
        self.port_type = port_type


class _Module:
    def __init__(self, title: str, category: ModuleCategory):
        self.metadata = _Metadata(title=title, category=category)
        self.input_ports: list[_Port] = []
        self.output_ports: list[_Port] = []

    def add_input(self, name: str) -> _Port:
        port = _Port(self, name, "input")
        self.input_ports.append(port)
        return port

    def add_output(self, name: str) -> _Port:
        port = _Port(self, name, "output")
        self.output_ports.append(port)
        return port

    def get_required_inputs(self) -> list[str]:
        return []

    def get_modulation_inputs(self) -> list[str]:
        return []


def _as_modules(*modules: _Module) -> list[AudioModule]:
    return cast(list[AudioModule], list(modules))


def _as_connections(
    *connections: tuple[_Port, _Port],
) -> list[tuple[PortWidget, PortWidget]]:
    return cast(list[tuple[PortWidget, PortWidget]], list(connections))


def test_patch_compiler_prevalidation_requires_output():
    source = _Module("Osc", ModuleCategory.SOURCE)
    source.add_output("Out")
    compiler = PatchCompiler()
    compiler.set_patch(_as_modules(source), _as_connections())
    errors = compiler.get_prevalidation_errors()
    assert "No output module in patch." in errors


def test_patch_compiler_prevalidation_detects_cycles():
    mod_a = _Module("A", ModuleCategory.MODIFIER)
    mod_a_out = mod_a.add_output("Out")
    mod_a_in = mod_a.add_input("In")
    mod_b = _Module("B", ModuleCategory.MODIFIER)
    mod_b_out = mod_b.add_output("Out")
    mod_b_in = mod_b.add_input("In")
    output = _Module("Output", ModuleCategory.OUTPUT)
    output_in = output.add_input("In")
    compiler = PatchCompiler()
    compiler.set_patch(
        _as_modules(mod_a, mod_b, output),
        _as_connections(
            (mod_a_out, mod_b_in), (mod_b_out, mod_a_in), (mod_a_out, output_in)
        ),
    )
    errors = compiler.get_prevalidation_errors()
    assert any("Infinite loop detected" in error for error in errors)


def test_patch_compiler_build_patch_tree_from_output():
    source = _Module("Osc", ModuleCategory.SOURCE)
    source_out = source.add_output("Out")
    output = _Module("Output", ModuleCategory.OUTPUT)
    output_in = output.add_input("In")
    compiler = PatchCompiler()
    compiler.set_patch(
        _as_modules(source, output), _as_connections((source_out, output_in))
    )
    tree = compiler.build_patch_tree()
    assert tree["name"] == "Output"
    assert tree["type"] == ModuleCategory.OUTPUT.value
    assert len(tree["inputs"]) == 1
    assert tree["inputs"][0]["port_name"] == "In"
    assert tree["inputs"][0]["node"]["name"] == "Osc"


def test_patch_compiler_build_patch_tree_without_output_creates_root():
    source = _Module("Osc", ModuleCategory.SOURCE)
    source.add_output("Out")
    compiler = PatchCompiler()
    compiler.set_patch(_as_modules(source), _as_connections())
    tree = compiler.build_patch_tree()
    assert tree["name"] == "Patch (no output)"
    assert tree["type"] == "ROOT"
    assert tree["inputs"][0]["node"]["name"] == "Osc"
