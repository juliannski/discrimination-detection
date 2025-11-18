# Discrimination Detection with Explanations

This repository accompanies the paper **"Discrimination Exposed? On the Reliability of Explanations for Discrimination Detection"** (Skirzynski, Danks & Ustun, 2025) accepted at FAccT 2025.

## Overview

This project provides code for running experiments to determine if users can detect discriminatory predictions with explanations. The web application allows researchers to conduct user studies examining the effectiveness of different explanation methods in identifying algorithmic discrimination.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Flask application:

```bash
python application.py
```

Then visit http://127.0.0.1:5000 to see the experiment interface.

## Results

The experimental results can be found in the `results/` directory:
- `results/experiments/` - Saved experiment handles (.pkl files)
- `results/results_single_explanation/` - Results for single explanation condition
- `results/results_multiple_explanations/` - Results for multiple explanations condition
- `results/results_shap_explanations/` - Results for SHAP explanations condition

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{skirzynski2025discrimination,
  title={Discrimination Exposed? On the Reliability of Explanations for Discrimination Detection},
  author={Skirzynski, Julian and Danks, David and Ustun, Berk},
  booktitle={ACM Conference on Fairness, Accountability, and Transparency (FAccT)},
  year={2025}
}
```

## Version

Current version: **v1.0.0**

## License

MIT License
