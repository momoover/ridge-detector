# Ridge Detector

A robust Python library for detecting and analyzing curvilinear structures in images using multi-scale ridge detection techniques. This package provides comprehensive tools for identifying, measuring, and analyzing ridge-like features across various scales, making it particularly useful for applications in medical imaging, remote sensing, and scientific image analysis.

## Features

- Multi-scale ridge detection with automatic scale selection
- Comprehensive width estimation of detected structures
- Junction detection and analysis
- Advanced network analysis of connected structures
- Detailed statistical analysis of detected features
- Interactive visualizations using Plotly
- Support for both dark and bright ridge detection
- Extensive configuration options for fine-tuning detection parameters

## Installation

TODO: Install via pip:

Or install from source:

```bash
git clone https://github.com/lxfhfut/ridge-detector.git
cd ridge-detector
pip install -e .
```

## Quick Start

```python
from ridge_detector import RidgeDetector

# Initialize detector with custom parameters
detector = RidgeDetector(
    low_contrast=150,
    high_contrast=255,
    min_len=50,
    dark_line=True,
    estimate_width=True
)

# Detect ridges in an image
detector.detect_lines("your_image.jpg")

# Save results with visualizations
detector.save_results(
    save_dir="output_directory",
    prefix="example",
    make_binary=True,
    draw_width=True
)
```

## Advanced Usage

### Detailed Analysis

```python
# Get point-by-point analysis
detailed_df = detector.save_detailed_results(save_dir="output")

# Compute network metrics
network_summary, node_metrics = detector.compute_network_analysis()

# Generate interactive visualizations
fig = detector.create_advanced_visualizations(
    save_dir="output",
    image_path="your_image.jpg"
)
```

### Customizing Detection Parameters

```python
detector = RidgeDetector(
    line_widths=np.arange(1, 5),  # Expected width range
    low_contrast=100,             # Lower bound for intensity contrast
    high_contrast=200,            # Upper bound for intensity contrast
    min_len=5,                    # Minimum length of detected ridges
    max_len=0,                    # Maximum length (0 for no limit)
    dark_line=True,               # True for dark ridges, False for bright
    estimate_width=True,          # Enable width estimation
    extend_line=False,            # Enable line extension at junctions
    correct_pos=False            # Enable position correction
)
```

## Output Types

1. **CSV Analysis Files**
   - Detailed point-by-point analysis
   - Statistical summaries
   - Network metrics
   - Shape characteristics

2. **Visualizations**
   - Contour overlays on original image
   - Width estimation visualization
   - Junction detection results
   - Interactive network graphs
   - Binary masks of detected structures

3. **Network Analysis**
   - Junction identification
   - Connectivity analysis
   - Node-level metrics
   - Network-level statistics

## API Reference

### Main Classes

#### RidgeDetector

The primary class for ridge detection and analysis.

```python
class RidgeDetector:
    def __init__(self,
                 line_widths=np.arange(1, 3),
                 low_contrast=100,
                 high_contrast=200,
                 min_len=5,
                 max_len=0,
                 dark_line=True,
                 estimate_width=True,
                 extend_line=False,
                 correct_pos=False)
```

### Key Methods

- `detect_lines(image)`: Performs ridge detection on the input image
- `save_detailed_results()`: Saves point-by-point analysis
- `compute_network_analysis()`: Computes network metrics
- `create_advanced_visualizations()`: Generates interactive visualizations
- `save_results()`: Saves all analysis results and visualizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Original Literature

This implementation is based on the following seminal works:

[1] Steger, Carsten. "An unbiased detector of curvilinear structures." IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(2), pp. 113-125, 1998.

[2] Lindeberg, Tony. "Edge detection and ridge detection with automatic scale selection." International Journal of Computer Vision, 30, pp. 117-156, 1998.

[3] Sato, Yoshinobu, et. al. "Three-dimensional multi-scale line filter for segmentation and visualization of curvilinear structures in medical images." Medical Image Analysis, 2(2), pp. 143-168, 1998.
