# AI-Powered Image Analysis & Edge Detection Platform

An interactive Streamlit portfolio project for secure image inspection, digital image processing, edge comparison, contour analysis, and transparent processing advice.

## Overview

Image-processing workflows often require several disconnected tools and opaque parameter choices. This application gives a user one in-memory workspace to upload a JPG or PNG, inspect its characteristics, apply safe preprocessing/filtering, compare five edge detectors, locate significant contours, and receive explainable rule-based suggestions.

## Objectives and features

- Validate and decode uploaded JPEG/PNG bytes safely; cap file size, pixel count, and dimensions.
- Report resolution, channels, brightness, contrast, RGB statistics, entropy, and histogram data.
- Provide grayscale conversion, normalization, resizing, equalization, CLAHE, and contrast enhancement modules.
- Compare mean, Gaussian, median, and bilateral filters using fixed safe kernels (3, 5, 7).
- Run Roberts, Prewitt, Sobel, Laplacian, and Canny edge detection with bounded settings.
- Calculate edge pixels, density, processing time, and relative edge strength; visualize comparisons in Plotly.
- Identify meaningful external contours, object count estimates, bounding rectangles, centers, areas, and perimeters.
- Calculate a clearly labelled heuristic quality score (0–100), including noise, sharpness, blur, brightness, contrast, and dynamic range.
- Download processed PNGs and CSV analysis/comparison reports.

## Architecture and processing pipeline

`app.py` is deliberately thin: it orchestrates Streamlit UI and cached results. Modules in `src/` handle image loading, analysis, preprocessing, filtering, edges, contours, quality, recommendations, and visualization.

Typical pipeline: **Upload → validation → grayscale → optional Gaussian/median filter → CLAHE → Canny/Sobel → contour detection → CSV/PNG export**.

## Technologies

Python 3.11+, Streamlit, OpenCV, NumPy, Pandas, Pillow, scikit-image, Plotly, and pytest. Dependencies are pinned in `requirements.txt`.

## Algorithms

Filtering uses mean, Gaussian, median, and bilateral filters. Edges use Roberts, Prewitt, Sobel, Laplacian, and Canny. Quality estimates use Laplacian variance for sharpness, robust high-frequency residuals for noise, and histogram percentile range for dynamic range.

## Intelligent recommendation engine

The “AI Recommendations” tab is a transparent, editable rule engine—not a trained AI or deep-learning model. It evaluates brightness, contrast, estimated noise, sharpness, edge density, dimensions, and heuristic quality to explain its suggested filter, enhancement, detector, and pipeline.

## Security and performance

The application never executes uploads, trusts file extensions, accepts paths from users, uses shell commands, or loads unsafe serialized data. It verifies decoded image content, rejects oversized inputs, bounds all user-controlled numeric parameters, processes in memory, and resizes large-but-acceptable images. Cached processing avoids duplicate grayscale and edge calculations during UI reruns. Normal users receive safe error messages rather than tracebacks.

## Project structure

```text
AI-Image-Analysis/
├── app.py                 # Streamlit UI/orchestration
├── src/                   # Modular processing code
├── tests/                 # Offline pytest suite
├── reports/               # Reserved for user-created reports
├── sample_images/         # Optional local samples
├── requirements.txt
└── .gitignore
```

## Installation and usage

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
streamlit run app.py
```

Upload a JPG/JPEG/PNG (up to 15 MB). Choose bounded filter/edge/contour options in the sidebar, inspect each tab, then use the download buttons. Run tests with `pytest`.

## Screenshots, results, and live demo

Add screenshots to this section after deployment. Results depend on the uploaded image and parameter selections. A live-demo URL is intentionally not claimed because none has been deployed.

## Limitations and future enhancements

The quality score is a practical heuristic rather than a universal scientific measurement; contours may merge or split depending on image content. Future work could add batch processing, user-saved parameter presets, additional color-space analysis, and benchmark datasets.

## GitHub and author

Ready to publish to GitHub as a portfolio repository. Add your repository URL, license, and author/contact details before publishing.
