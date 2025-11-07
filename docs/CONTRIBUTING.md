<div align="center">
  <img src="internal/gahan_logo.png" alt="Gahan AI Logo" width="150"/>
  
  # Contributing Guidelines
  
  **FBMC vs OFDM Validation Framework**
</div>

---

Thank you for your interest in contributing to the FBMC vs OFDM Validation Framework! This document provides guidelines for contributing to this scientific research project.

## 🎯 Project Scope

This framework is designed for:
- **Academic Research**: Validating theoretical predictions about FBMC vs OFDM
- **Reproducible Science**: Providing standardized comparison methodology
- **Educational Use**: Demonstrating advanced signal processing concepts
- **Industry Reference**: Benchmarking waveform performance

## 🚀 Quick Start for Contributors

### 1. Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/fbmc_ttdf.git
cd fbmc_ttdf

# Install with development dependencies
uv sync --dev

# Verify installation
uv run python -m pytest
```

### 2. Make Your Changes

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# Test your changes
uv run main.py

# Run tests
uv run python -m pytest
```

### 3. Submit Your Contribution

```bash
# Format code
uv run black .
uv run ruff check --fix .

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push and create pull request
git push origin feature/your-feature-name
```

## 📝 Types of Contributions

### 🔬 Scientific Contributions

#### Algorithm Improvements
- **New Waveforms**: Implement additional waveforms (UFMC, GFDM, etc.)
- **Enhanced Processing**: Improve radar processing algorithms
- **Filter Design**: Better prototype filters for FBMC
- **Channel Models**: Add realistic channel conditions

#### Performance Metrics
- **New Metrics**: Additional performance measures
- **Statistical Analysis**: Confidence intervals, hypothesis testing
- **Complexity Analysis**: FLOP counting, memory usage
- **Hardware Metrics**: Power consumption, silicon area

#### Validation Enhancements
- **Extended Scenarios**: More realistic test cases
- **Parameter Sweeps**: Systematic parameter analysis
- **Cross-validation**: MATLAB vs Python consistency
- **Benchmarking**: Against reference implementations

### 💻 Technical Contributions

#### Code Quality
- **Performance Optimization**: Faster algorithms, vectorization
- **Memory Efficiency**: Reduced memory footprint
- **Error Handling**: Robust error detection and recovery
- **Code Documentation**: Improved docstrings and comments

#### Features
- **Interactive Plots**: Jupyter notebook integration
- **Configuration**: YAML/JSON configuration files
- **Export Options**: Additional output formats
- **Parallel Processing**: Multi-threading support

#### Testing
- **Unit Tests**: Individual function testing
- **Integration Tests**: End-to-end validation
- **Performance Tests**: Regression testing
- **Reproducibility Tests**: Cross-platform verification

### 📚 Documentation Contributions

#### User Documentation
- **Tutorials**: Step-by-step guides
- **Examples**: Additional use cases
- **FAQ**: Common questions and solutions
- **Video Tutorials**: Screen recordings

#### Developer Documentation  
- **Architecture**: System design documentation
- **API Reference**: Complete function documentation
- **Development Guides**: Setup and workflow
- **Contribution Examples**: Sample contributions

## 🔧 Development Guidelines

### Code Style

We use automated tools for consistent code formatting:

```bash
# Format Python code
uv run black .

# Lint and fix issues
uv run ruff check --fix .

# Type checking (future)
uv run mypy .
```

#### Python Style Guidelines
- **PEP 8**: Follow Python style guide
- **Type Hints**: Add type annotations for new functions
- **Docstrings**: Use NumPy-style docstrings
- **Variable Names**: Descriptive names, avoid abbreviations

```python
def compute_spectral_efficiency(
    signal: np.ndarray, 
    cyclic_prefix_ratio: float
) -> float:
    """
    Compute spectral efficiency of a waveform.
    
    Parameters
    ----------
    signal : np.ndarray
        Time-domain signal samples
    cyclic_prefix_ratio : float
        Ratio of cyclic prefix to useful symbol duration
        
    Returns
    -------
    float
        Spectral efficiency in bits/s/Hz
        
    Examples
    --------
    >>> efficiency = compute_spectral_efficiency(ofdm_signal, 0.25)
    >>> print(f"Efficiency: {efficiency:.3f}")
    """
```

### Testing Standards

#### Test Categories
1. **Unit Tests**: Individual function validation
2. **Integration Tests**: Component interaction
3. **Regression Tests**: Performance consistency
4. **Scientific Tests**: Theoretical validation

#### Test Structure
```python
import pytest
import numpy as np
from fbmc_ofdm_validation import OFDM, FBMC

class TestWaveformGeneration:
    def test_ofdm_symbol_generation(self):
        """Test OFDM symbol generation."""
        ofdm = OFDM(N_subcarriers=64)
        symbols = ofdm.generate_qam_symbols(M=16, num_symbols=10)
        
        assert symbols.shape == (64, 10)
        assert np.allclose(np.mean(np.abs(symbols)**2), 1.0, rtol=0.1)
    
    def test_fbmc_modulation_length(self):
        """Test FBMC signal length."""
        fbmc = FBMC(N_subcarriers=64, K_overlapping=4)
        symbols = fbmc.generate_qam_symbols(M=16, num_symbols=10)
        signal = fbmc.modulate(symbols)
        
        expected_length = (20 + 4 - 1) * 64 // 2  # OQAM symbols
        assert len(signal) == expected_length
```

### Scientific Rigor

#### Validation Requirements
- **Theoretical Consistency**: Results should match theory
- **Cross-Platform**: Identical results on different systems
- **Parameter Sensitivity**: Document parameter dependencies
- **Statistical Significance**: Use appropriate sample sizes

#### Documentation Standards
- **Mathematical Notation**: Use LaTeX for equations
- **Reference Papers**: Cite relevant literature
- **Algorithm Description**: Clear step-by-step explanations
- **Assumptions**: Document limitations and assumptions

```python
def compute_ambiguity_function(signal, delays, dopplers):
    """
    Compute waveform ambiguity function.
    
    The ambiguity function is defined as:
    
    .. math::
        \chi(\tau, f_d) = \int_{-\infty}^{\infty} s(t) s^*(t-\tau) 
                          e^{-j2\pi f_d t} dt
    
    References
    ----------
    .. [1] Levanon, N., & Mozeson, E. (2004). Radar signals. 
           John Wiley & Sons.
    """
```

## 🧪 Research Contributions

### Adding New Waveforms

To add a new waveform type:

1. **Create Waveform Class**:
```python
class UFMC(WaveformGenerator):
    def __init__(self, N_subcarriers=256, filter_length=43):
        super().__init__(N_subcarriers, 0)
        self.filter_length = filter_length
        self.prototype_filter = self.design_dolph_chebyshev_filter()
    
    def modulate(self, symbols):
        # Implementation here
        pass
```

2. **Add to Analysis Functions**:
```python
def plot_three_way_comparison():
    """Compare OFDM, FBMC, and UFMC."""
    ofdm = OFDM()
    fbmc = FBMC()
    ufmc = UFMC()
    # Implementation here
```

3. **Update Tests**:
```python
def test_ufmc_implementation():
    """Validate UFMC implementation."""
    # Test cases here
```

### Adding New Metrics

To add a new performance metric:

1. **Implement Metric Function**:
```python
def compute_bit_error_rate(tx_bits, rx_bits):
    """
    Compute bit error rate.
    
    Parameters
    ----------
    tx_bits : np.ndarray
        Transmitted bits
    rx_bits : np.ndarray
        Received bits
        
    Returns
    -------
    float
        Bit error rate
    """
    errors = np.sum(tx_bits != rx_bits)
    return errors / len(tx_bits)
```

2. **Add to Metrics Function**:
```python
def compute_performance_metrics():
    # Existing metrics...
    
    # Add new metric
    ber_ofdm = compute_bit_error_rate(tx_bits, rx_bits_ofdm)
    ber_fbmc = compute_bit_error_rate(tx_bits, rx_bits_fbmc)
    
    metrics['Metric'].append('Bit Error Rate')
    metrics['OFDM'].append(f'{ber_ofdm:.2e}')
    metrics['FBMC'].append(f'{ber_fbmc:.2e}')
    # etc.
```

## 🐛 Bug Reports

### Before Reporting
1. **Check Existing Issues**: Search for similar problems
2. **Reproduce**: Verify the bug is reproducible
3. **Minimal Example**: Create smallest failing case
4. **Environment**: Document system details

### Bug Report Template
```markdown
## Bug Description
Brief description of the issue.

## Steps to Reproduce
1. Run command: `uv run main.py`
2. Expected: Normal execution
3. Actual: Error message

## Environment
- OS: Ubuntu 22.04
- Python: 3.12.0
- Dependencies: (output of `uv tree`)

## Error Output
```
[Error traceback here]
```

## Additional Context
Any other relevant information.
```

## 💡 Feature Requests

### Feature Request Template
```markdown
## Feature Description
Clear description of the proposed feature.

## Scientific Justification
Why is this feature scientifically valuable?

## Implementation Approach
Suggested approach or algorithm.

## Alternatives Considered
Other approaches you've considered.

## References
Relevant papers or standards.
```

## 📋 Review Process

### Pull Request Requirements
- [ ] **Code Quality**: Passes linting and formatting
- [ ] **Tests**: New code has appropriate tests
- [ ] **Documentation**: Functions are documented
- [ ] **Scientific Validation**: Results are theoretically sound
- [ ] **Backwards Compatibility**: Doesn't break existing functionality

### Review Criteria

#### Scientific Review
- **Correctness**: Algorithm implementation matches theory
- **Performance**: Computational efficiency is reasonable
- **Accuracy**: Numerical precision is appropriate
- **Completeness**: Edge cases are handled

#### Code Review
- **Readability**: Code is clear and well-commented
- **Maintainability**: Code is modular and extensible
- **Performance**: No obvious inefficiencies
- **Testing**: Adequate test coverage

### Review Timeline
- **Initial Response**: 2-3 days
- **Full Review**: 1-2 weeks
- **Revisions**: As needed
- **Merge**: After approval from maintainers

## 🏷️ Release Process

### Version Numbering
We use semantic versioning (SemVer):
- **Major**: Breaking changes to API
- **Minor**: New features, backwards compatible
- **Patch**: Bug fixes, documentation updates

### Release Workflow
1. **Feature Freeze**: No new features
2. **Testing**: Comprehensive validation
3. **Documentation**: Update all docs
4. **Tag Release**: Create git tag
5. **Announce**: Update README and notifications

## 🤝 Community Guidelines

### Code of Conduct
- **Respectful**: Be considerate in all interactions
- **Constructive**: Provide helpful feedback
- **Inclusive**: Welcome contributors from all backgrounds
- **Scientific**: Focus on technical merit

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions and discussions
- **Discussions**: General questions and ideas

### Recognition
Contributors are recognized in:
- **AUTHORS.md**: List of all contributors
- **Release Notes**: Highlighting major contributions
- **Academic Citations**: Co-authorship for significant contributions

## 📖 Resources

### Learning Resources
- **Signal Processing**: [scipy.signal documentation](https://docs.scipy.org/doc/scipy/reference/signal.html)
- **FBMC Theory**: Bellanger, M. (2012). FBMC physical layer: a primer
- **Radar Processing**: Richards, M. A. (2014). Fundamentals of radar signal processing
- **Python Scientific**: [NumPy documentation](https://numpy.org/doc/)

### Development Tools
- **uv**: [Package manager documentation](https://docs.astral.sh/uv/)
- **pytest**: [Testing framework](https://docs.pytest.org/)
- **matplotlib**: [Plotting library](https://matplotlib.org/stable/contents.html)
- **Git**: [Version control guide](https://git-scm.com/doc)

## 🆘 Getting Help

### For Contributors
1. **Read Documentation**: Start with README and API docs
2. **Check Examples**: Look at existing implementations
3. **Ask Questions**: Use GitHub Discussions
4. **Join Community**: Connect with other contributors

### Contact
- **Company**: [Gahan AI Private Limited](https://gahanai.com)
- **Maintainer**: Research Team ([research@gahanai.com](mailto:research@gahanai.com))
- **GitHub Issues**: Technical problems
- **Discussions**: General questions
- **Email**: Academic collaboration

---

<div align="center">
  <img src="internal/gahan_logo.png" alt="Gahan AI" width="80"/>
  <br>
  <strong>Powered by Gahan AI Research</strong>
  <br>
  <sub>Thank you for contributing to advancing joint radar-communication research!</sub>
</div>