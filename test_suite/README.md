# Morphological parameters test suite

<!-- ---

### Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Resources](#resources)

--- -->

## Overview

Here we test the morphological parameters calculated by `statmorph-lsst` and `Galfit` for our set of 167 NGC/IC galaxies observed with HST F814W. The procedure is as follows:

1. Estimate the original image background noise level with iterative masking
2. Make cutouts of raw data files and scale them to at least 25 pc/px resolution
3. Create segmentation maps detecting the main galaxy and masking other sources
4. Create augmented images for a range of noise / resolution levels
5. Run `statmorph-lsst` and `Galfit` on all augmented images to get morphological parameters

---

<!-- 
This project provides advanced morphological analysis tools specifically designed for Large Survey of Space and Time (LSST) astronomical data. The package extends the capabilities of the original statmorph library to handle the unique requirements and data formats of LSST observations.

### Key Objectives

- **High-performance analysis** of galaxy morphologies
- **Seamless integration** with LSST data pipelines
- **Robust statistical measures** for large-scale surveys
- **Comprehensive documentation** and examples

---

## Features

### Core Functionality

• **Morphological Parameters**
  - Gini coefficient and M20 moment
  - Concentration, asymmetry, and smoothness (CAS)
  - Sérsic profile fitting
  - Petrosian radii calculations

• **LSST-Specific Features**
  - Native support for LSST data formats
  - Optimized for large-scale processing
  - Integration with LSST Science Pipelines
  - Multi-band analysis capabilities

• **Advanced Analysis Tools**
  - Statistical uncertainty estimation
  - Batch processing capabilities
  - Diagnostic plotting utilities
  - Quality assessment metrics

---

## Installation

### Prerequisites

1. **Python Environment**
   - Python 3.8 or higher
   - NumPy >= 1.18.0
   - SciPy >= 1.5.0
   - Matplotlib >= 3.2.0

2. **LSST Dependencies**
   - LSST Science Pipelines
   - lsst.afw
   - lsst.pipe.base

3. **Optional Dependencies**
   - Jupyter Notebooks for examples
   - Astropy for additional astronomical utilities

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/astro-nova/statmorph-lsst.git
cd statmorph-lsst

# Install in development mode
pip install -e .

# Install optional dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Basic Usage Example

```python
import statmorph
import numpy as np
from lsst.afw.image import ExposureF

# Load your LSST exposure
exposure = ExposureF("path/to/your/exposure.fits")

# Extract image data
image = exposure.getMaskedImage().getImage().getArray()

# Perform morphological analysis
source_morphs = statmorph.source_morphology(
    image, 
    segmap, 
    gain=1.0, 
    psf=psf_image
)

# Access results
print(f"Gini coefficient: {source_morphs.gini}")
print(f"M20: {source_morphs.m20}")
print(f"Concentration: {source_morphs.concentration}")
```

---

## Examples

### 1. Single Galaxy Analysis

![Single Galaxy Example](https://via.placeholder.com/600x400/4A90E2/FFFFFF?text=Single+Galaxy+Analysis)

Detailed example of analyzing a single galaxy's morphological properties.

### 2. Batch Processing

![Batch Processing Workflow](https://via.placeholder.com/600x400/7ED321/FFFFFF?text=Batch+Processing+Workflow)

Demonstration of processing multiple galaxies efficiently using parallel computing.

### 3. Multi-band Analysis

![Multi-band Results](https://via.placeholder.com/600x400/F5A623/FFFFFF?text=Multi-band+Analysis+Results)

Example showing how morphological parameters vary across different photometric bands.

---

## Documentation

### API Reference

- **[Core Functions](docs/api.rst)** - Complete API documentation
- **[Tutorial Notebooks](docs/notebooks/)** - Interactive examples
- **[Parameter Descriptions](docs/description.rst)** - Detailed parameter explanations

### User Guides

1. **Getting Started Guide**
   - Installation instructions
   - Basic configuration
   - First analysis example

2. **Advanced Features**
   - Custom PSF handling
   - Uncertainty estimation
   - Performance optimization

3. **Integration with LSST**
   - Pipeline integration
   - Data format handling
   - Best practices

---

## Contributing

We welcome contributions from the astronomical community! Here's how you can help:

### Ways to Contribute

- 🐛 **Report bugs** and suggest improvements
- 📖 **Improve documentation** and add examples
- 🔬 **Add new morphological parameters** or analysis methods
- 🚀 **Optimize performance** for large datasets
- 🧪 **Write tests** to improve code reliability

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) conventions
- Use descriptive variable names
- Add docstrings for all functions and classes
- Include type hints where appropriate

---

## Resources

### External Links

- **[LSST Science Pipelines](https://pipelines.lsst.io/)** - Official LSST pipeline documentation
- **[Original Statmorph](https://statmorph.readthedocs.io/)** - Base package documentation
- **[LSST Data Products](https://www.lsst.org/about/dm)** - Information about LSST data formats
- **[Galaxy Morphology Papers](https://ui.adsabs.harvard.edu/search/q=galaxy%20morphology)** - Relevant literature on ADS

### Community

- **[LSST Community Forum](https://community.lsst.org/)** - Ask questions and share ideas
- **[GitHub Issues](https://github.com/astro-nova/statmorph-lsst/issues)** - Report bugs and request features
- **[Discussions](https://github.com/astro-nova/statmorph-lsst/discussions)** - General discussion and Q&A

### Citation

If you use this package in your research, please cite:

```bibtex
@software{statmorph_lsst,
  author = {Author Name and Contributors},
  title = {Statmorph-LSST: Morphological Analysis for LSST Data},
  url = {https://github.com/astro-nova/statmorph-lsst},
  version = {1.0.0},
  year = {2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The LSST Project for providing the foundational data analysis framework
- The original statmorph development team
- The astronomical community for valuable feedback and contributions

---

*Last updated: October 24, 2025*

![Project Logo](https://via.placeholder.com/200x100/8E44AD/FFFFFF?text=Statmorph+LSST) -->
