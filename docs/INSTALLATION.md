<div align="center">
  <img src="internal/gahan_logo.png" alt="Gahan AI Logo" width="150"/>
  
  # Installation Guide
  
  **FBMC vs OFDM Validation Framework**
</div>

---

## Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: Version 3.12 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large parameter sweeps)
- **Storage**: ~100MB for installation, ~1GB for generated plots and data

### Required Software

#### 1. Python 3.12+
```bash
# Check Python version
python --version

# If Python 3.12+ is not installed:
# Ubuntu/Debian
sudo apt update && sudo apt install python3.12 python3.12-venv

# macOS (using Homebrew)
brew install python@3.12

# Windows: Download from python.org
```

#### 2. uv Package Manager (Recommended)
```bash
# Install uv (cross-platform)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv

# Verify installation
uv --version
```

## Installation Methods

### Method 1: Using uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd fbmc_ttdf

# uv automatically manages dependencies
uv run main.py
```

The first run will automatically:
1. Create a virtual environment
2. Install all required dependencies
3. Execute the validation framework

### Method 2: Traditional pip Installation

```bash
# Clone the repository
git clone <repository-url>
cd fbmc_ttdf

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the framework
python main.py
```

### Method 3: Development Installation

For development and customization:

```bash
# Clone repository
git clone <repository-url>
cd fbmc_ttdf

# Install in development mode with extra tools
uv sync --dev

# Run with development dependencies
uv run python main.py
```

## Dependencies

### Core Dependencies (Auto-installed)

```toml
numpy >= 2.3.4          # Numerical computing
matplotlib >= 3.10.7    # Plotting and visualization
scipy >= 1.16.3         # Signal processing
```

### Development Dependencies (Optional)

```toml
pytest >= 7.0.0         # Testing framework
black >= 23.0.0         # Code formatting
ruff >= 0.1.0           # Linting and code quality
```

## Verification

### Quick Test
```bash
cd fbmc_ttdf
uv run main.py
```

Expected output:
```
============================================================
FBMC vs OFDM VALIDATION FRAMEWORK
For IEEE Paper: FBMC Waveforms for Joint Radar-Communication
============================================================
Generating Spectral Comparison...
Generating Range-Doppler Comparison...
...
✓ All validation plots saved successfully!
```

### Check Generated Files
```bash
ls -la logs/
```

Should show:
- `spectral_comparison.png`
- `range_doppler_comparison.png`
- `ambiguity_comparison.png`
- `doppler_tolerance.png`
- `metrics_table.png`

## Troubleshooting

### Common Issues

#### 1. Python Version Error
```bash
ERROR: Python 3.12+ required
```
**Solution**: Upgrade Python or use pyenv:
```bash
pyenv install 3.12.0
pyenv local 3.12.0
```

#### 2. Memory Error During Execution
```bash
MemoryError: Unable to allocate array
```
**Solution**: Reduce system parameters in the code:
```python
# In fbmc_ofdm_validation.py, reduce:
N_subcarriers = 128  # instead of 256
nfft = 4096         # instead of 8192
```

#### 3. Display Issues (Linux)
```bash
UserWarning: Matplotlib is currently using agg
```
**Solution**: Install GUI backend:
```bash
# Ubuntu/Debian
sudo apt install python3-tk

# Or set backend in code
import matplotlib
matplotlib.use('Agg')  # For headless systems
```

#### 4. Permission Errors
```bash
PermissionError: [Errno 13] Permission denied: './logs/'
```
**Solution**: Check directory permissions:
```bash
chmod 755 logs/
# Or run from user directory
```

#### 5. Import Errors
```bash
ModuleNotFoundError: No module named 'numpy'
```
**Solution**: Verify virtual environment:
```bash
# Check if in correct environment
which python
# Should show path with fbmc_ttdf

# Reinstall dependencies
uv sync --reinstall
```

### Platform-Specific Issues

#### Windows
- Use PowerShell or Command Prompt
- Path separators: Use forward slashes or escape backslashes
- Long path issues: Enable long path support in Group Policy

#### macOS
- May need Xcode command line tools: `xcode-select --install`
- Use Homebrew for Python installation
- Check security settings for script execution

#### Linux
- Install system-level dependencies for matplotlib
- May need to install tkinter separately
- Check Python development headers: `python3-dev`

## Advanced Installation

### Custom Configuration

Create `config.py` for custom parameters:
```python
# config.py
CUSTOM_PARAMS = {
    'N_subcarriers': 512,
    'sample_rate': 2e9,
    'output_dir': './custom_logs/',
}
```

### Docker Installation

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv sync

CMD ["uv", "run", "main.py"]
```

```bash
# Build and run
docker build -t fbmc-validation .
docker run -v $(pwd)/logs:/app/logs fbmc-validation
```

### HPC/Cluster Installation

For high-performance computing environments:

```bash
# Load modules (cluster-specific)
module load python/3.12
module load gcc/11.2.0

# Install in user space
pip install --user uv
uv run main.py
```

## Performance Optimization

### Memory Optimization
```python
# For limited memory systems
import gc
gc.collect()  # Add periodic garbage collection
```

### Parallel Processing
```python
# Enable numpy multithreading
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### Storage Optimization
```bash
# Compress generated plots
find logs/ -name "*.png" -exec pngquant --ext .png --force {} \;
```

## IDE Integration

### VS Code
1. Install Python extension
2. Select Python interpreter in virtual environment
3. Configure debug settings:

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "FBMC Validation",
            "type": "python",
            "request": "launch",
            "program": "main.py",
            "console": "integratedTerminal"
        }
    ]
}
```

### PyCharm
1. Open project directory
2. Configure Python interpreter to use virtual environment
3. Set run configuration for `main.py`

### Jupyter Integration
```bash
# Install Jupyter
uv add jupyter

# Create notebook kernel
uv run python -m ipykernel install --name fbmc-env --display-name "FBMC Validation"

# Start Jupyter
uv run jupyter lab
```

## Next Steps

After successful installation:

1. **Run Basic Validation**: `uv run main.py`
2. **Examine Output**: Check `logs/` directory for generated plots
3. **Read Documentation**: Review `docs/API.md` for customization
4. **Explore Parameters**: Modify system parameters for different scenarios
5. **Contribute**: See development guidelines for enhancements

## Support

For installation issues:
1. Check this troubleshooting guide
2. Verify system requirements
3. Check [GitHub Issues](https://github.com/codesteller/drwig-jrc-fbmc/issues)
4. Contact [research@gahanai.com](mailto:research@gahanai.com)

---

<div align="center">
  <img src="internal/gahan_logo.png" alt="Gahan AI" width="80"/>
  <br>
  <strong>© 2025 Gahan AI Private Limited</strong>
</div>
4. Create new issue with system details and error messages