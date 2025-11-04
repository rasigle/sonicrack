# Contributing to AudioPlayground

Thank you for your interest in contributing to AudioPlayground! This document provides guidelines for contributing to the project.

## 🎯 Ways to Contribute

- **Report Bugs**: Submit detailed bug reports
- **Suggest Features**: Propose new features or improvements
- **Write Documentation**: Improve docs, tutorials, examples
- **Submit Code**: Fix bugs, implement features, optimize performance
- **Create Modules**: Build custom synthesizer modules
- **Share Presets**: Contribute interesting patches

## 🚀 Getting Started

### 1. Set Up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/AudioPlayground.git
cd AudioPlayground

# Install development dependencies
pip install -e ".[dev]"

# Run tests to verify setup
python -m pytest tests/ -v
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 3. Make Your Changes

Follow the coding standards below and ensure tests pass.

## 📋 Coding Standards

### Python Style

- **PEP 8** compliance
- **Type hints** for all functions/methods
- **Google-style docstrings**
- **Maximum line length**: 88 characters (Black formatter)

### Example

```python
def process_audio(
    samples: np.ndarray,
    gain: float = 1.0,
    sample_rate: int = 44100
) -> np.ndarray:
    """Process audio samples with gain adjustment.
    
    Args:
        samples: Input audio samples as numpy array
        gain: Gain multiplier (default: 1.0)
        sample_rate: Sample rate in Hz (default: 44100)
    
    Returns:
        Processed audio samples
        
    Raises:
        ValueError: If gain is negative
    """
    if gain < 0:
        raise ValueError("Gain must be non-negative")
    return samples * gain
```

### Code Organization

- **Engine code**: `src/engine/`
- **GUI code**: `src/gui/`
- **Builder API**: `src/builder/`
- **Tests**: `tests/` (mirror source structure)
- **Examples**: `examples/`
- **Documentation**: `docs/`

### Testing

- Write tests for all new features
- Maintain 95%+ test coverage
- Run full test suite before submitting:

```bash
python -m pytest tests/ -v
```

### Performance

- Use NumPy vectorization for audio processing
- Profile performance-critical code
- Aim for 100x+ realtime performance

## 🐛 Reporting Bugs

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**Steps to Reproduce**
1. Step one
2. Step two
3. ...

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 11]
- Python: [e.g., 3.11.5]
- AudioPlayground: [version or commit]

**Additional Context**
Error messages, screenshots, etc.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why is this feature needed?

**Proposed Implementation**
How might this be implemented?

**Alternatives Considered**
Other ways to achieve the same goal
```

## 🔧 Pull Request Process

### 1. Before Submitting

- [ ] Tests pass (`pytest tests/`)
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Changelog entry (if applicable)

### 2. PR Description Template

```markdown
**What does this PR do?**
Brief description

**Why is this needed?**
Context and motivation

**How was this tested?**
Testing approach

**Related Issues**
Closes #123
```

### 3. Review Process

- Maintainers will review within 3-5 days
- Address feedback and update PR
- Once approved, PR will be merged

## 🎨 Creating Custom Modules

### Engine Component

```python
from src.engine.audio_component import AudioComponent, ComponentDescriptor

@register_component
class MyOscillator(Oscillator):
    descriptor = ComponentDescriptor(
        name="MyOscillator",
        category=ComponentCategory.OSCILLATOR,
        config_params=["frequency", "amplitude"],
        description="Custom oscillator"
    )
    
    def __next__(self):
        # Generate next sample
        return self.sample_value
```

### GUI Module

```python
from src.gui.module_registry import register_module
from src.gui.widgets.module_widget import ModuleWidget

@register_module()
class MyModuleWidget(ModuleWidget):
    @property
    def module_title(self) -> str:
        return "My Module"
    
    @property
    def module_category(self) -> ModuleCategory:
        return ModuleCategory.SOURCE
```

See [Plugin Development Guide](../docs/developer/plugin-development.md) for details.

## 📝 Documentation

### Writing Style

- Clear and concise
- Include code examples
- Use screenshots for UI features
- Update table of contents

### Building Docs

```bash
# Preview documentation
# (Add mkdocs or sphinx when implemented)
```

## 🎯 Areas Needing Help

**High Priority**:
- Filter modules (low-pass, high-pass, band-pass)
- MIDI input support
- Additional effects (reverb, delay, chorus)
- More example presets
- Tutorial videos

**Medium Priority**:
- Wavetable synthesis
- Polyphony system
- Performance optimizations
- UI improvements (zoom, undo/redo)

**Low Priority**:
- Plugin marketplace
- Cloud preset sharing
- Mobile/tablet support

## 💬 Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- Be respectful and considerate
- Accept constructive criticism gracefully
- Focus on what's best for the project
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information

### Enforcement

Violations may result in temporary or permanent ban from the project.

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Thanked in project updates

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to AudioPlayground!** 🎵

Questions? Open a GitHub Discussion or contact the maintainers.

