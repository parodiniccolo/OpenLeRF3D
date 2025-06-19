# OpenLeRF3D: Open-Vocabulary 3D Scene Understanding with LeRF Guidance

OpenLeRF3D is an interactive tool that enables natural language querying of 3D scenes using Language-Embedded Radiance Fields (LeRF). Built on top of Nerfstudio, it allows users to locate and visualize objects or regions in Neural Radiance Field reconstructions by describing them in plain text.



## Features

- **Natural Language 3D Querying**: Find objects and regions in 3D scenes using text descriptions
- **Multiple Visualization Modes**:
  - **Heatmap**: Color-coded confidence visualization across all points
  - **Filtered Heatmap**: Shows only high-confidence points with heatmap coloring
  - **Mixed Colors**: Preserves original scene colors while highlighting matches
  - **Filtered Original Colors**: High-confidence points with their original scene colors
- **Interactive Training Pipeline**: Integrated LeRF training directly from the interface
- **Advanced Visualization Options**:
  - Confidence threshold adjustment
  - Global min-max normalization
  - Point cloud densification for better visualization
  - Sphere representation for enhanced viewing
- **Nerfstudio Compatibility**: Works with any Nerfstudio-compatible dataset
- **Real-time Results**: Interactive Gradio interface with live 3D visualization
- **Export Capabilities**: Download query results as PLY point clouds

## Installation

Create a conda environment:

```bash
conda create --name openlerf3d -y python=3.10
conda activate openlerf3d
pip install --upgrade pip
```

Install all requirements using the provided script:

```bash
bash install_requirements.sh
```

## Usage

### 1. Launch the Interface

```bash
python gradio_interface.py
```

### 2. Configure Your Scene

- **Scene Name**: Enter a name for your scene
- **NeRF Studio Data Path**: Path to your Nerfstudio-compatible dataset
- **Work Directory**: Directory where outputs will be saved
- **LERF Type**: Choose from `lerf-big`, `lerf`, `depthlerf-big`, or `depthlerf`

### 3. Run Training Pipeline

Click "Run Training Pipeline" to:
- Train the LeRF model on your scene
- Export point clouds and embeddings
- Prepare the scene for querying

### 4. Initialize Scene

After training completes, click "Initialize Scene" to load the trained model and prepare for queries.

### 5. Query Your Scene

- Enter natural language descriptions (e.g., "window", "door", "red chair")
- Adjust confidence threshold for filtering results
- Choose visualization mode
- Enable enhancements like normalization or sphere representation

## Data Format

OpenLeRF3D works with any Nerfstudio-compatible dataset structure:

```
your_scene/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── transforms.json
```

Supported input formats include:
- COLMAP reconstructions
- Instant-NGP format
- Record3D captures
- Custom Nerfstudio formats

## Visualization Modes

- **Heatmap**: All points colored by confidence scores using a color gradient
- **Filtered Heatmap**: Only points above threshold shown with heatmap colors
- **Mixed Colors**: Original scene colors preserved, matches highlighted with heatmap
- **Filtered Original Colors**: High-confidence points shown with original scene colors

## Configuration Options

- **Confidence Threshold**: Filter results based on match confidence (0.1-0.9)
- **Normalization**: Apply global min-max normalization to confidence scores
- **Sphere Representation**: Render points as spheres for better visualization
- **Density Factor**: Increase point density for enhanced visual quality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LERF](https://github.com/kerrj/lerf) - Language Embedded Radiance Fields
- [Depth-LERF](https://github.com/parodiniccolo/depth-lerf) - Language Embedded Radiance Fields with Depth Supervision
- [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) - Neural Radiance Field framework
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - Open source CLIP implementation
