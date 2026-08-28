# DeepCPR

Deep learning-based Chromatographic Profile Resolution

## Overview

DeepCPR is a tool for automatic resolution of complex GC-MS data based on
chromatographic profile prediction. It divides raw GC-MS data into segments,
predicts chromatographic profiles, and estimates the corresponding mass
spectra using iterative multivariate curve resolution (MCR) methods. The
workflow produces resolved peak tables and mass spectra for downstream
compound identification and statistical analysis.

The repository also includes an OPLS-DA workflow for discriminant analysis and
biomarker screening. Resolved spectra can be searched against the NIST library
or converted for use with [FastEI](https://github.com/Qiong-Yang/FastEI/tree/main).

<div align="center">
<img src="https://github.com/YuChuanxiu/DeepCPR/blob/main/workflow.png" width="785" alt="DeepCPR workflow" />
</div>

## Key features

- Automatic resolution of complex GC-MS chromatographic profiles.
- Export of peak tables, segment information, and NIST-compatible MSP files.
- Optional generation of chromatographic resolution figures.
- Adaptive resolution for segments containing more than five co-eluting
  components.
- TensorFlow/H5 and TensorFlow-independent ONNX Runtime inference paths.
- A Gradio-based local browser interface for users who prefer a graphical
  workflow.

## Installation

We recommend using [Conda](https://conda.io/docs/user-guide/install/download.html)
and [pip](https://pypi.org/project/pip/). Python 3.10 is supported.

Clone the repository and enter its root directory:

```bash
git clone https://github.com/YuChuanxiu/DeepCPR.git
cd DeepCPR
```

Choose one of the following inference environments:

| Runtime | Model format | Installation |
|---|---|---|
| TensorFlow | `.h5` | `pip install -r requirements.txt` |
| ONNX Runtime | `.onnx` | `pip install -r requirements-onnx.txt` |

The TensorFlow environment reproduces the original H5-based workflow. 
The ONNX environment supports inference without requiring TensorFlow, and it can be deployed on other major platforms such as PyTorch.

## Pretrained models and example data

Download the pretrained models and example datasets from
[Release v1.1.0](https://github.com/YuChuanxiu/DeepCPR/releases/tag/v1.1.0):

- `DeepCPR.h5` and `DeepCS.h5` for the TensorFlow/H5 workflow;
- `DeepCPR.onnx` and `DeepCS.onnx` for TensorFlow-independent inference;
- `data.zip` containing example GC-MS datasets.

Place the model files in the `example` directory and extract `data.zip` there.
The resulting directory structure should be:

```text
DeepCPR/
├── example/
│   ├── data/
│   ├── DeepCPR.h5
│   ├── DeepCS.h5
│   ├── DeepCPR.onnx       # optional: ONNX Runtime workflow
│   └── DeepCS.onnx        # optional: ONNX Runtime workflow
└── example.ipynb
```

## Quick start: local graphical interface

DeepCPR provides a Gradio-based local browser interface for users who prefer
not to use the command line. All data processing and model inference are
performed on the user's local machine; raw GC-MS data are not uploaded to an
external server.

From the repository root, launch the interface with:

```bash
python DeepCPR/app.py
```

The interface opens automatically at:

```text
http://127.0.0.1:7860
```

The interface supports:

- uploading multiple GC-MS files in CDF or NetCDF format;
- selecting DeepCS and DeepCPR models in H5 or ONNX format;
- enabling adaptive resolution and optional figure generation;
- monitoring processing status and per-file runtime;
- previewing peak tables, segment information, and resolution figures;
- downloading all outputs as a ZIP archive.

The output archive may contain:

- `peak_area_table.csv`: merged peak table;
- `single/*.csv`: peak tables for individual files;
- `seg/*.csv`: segment information;
- `ms/**/*.msp`: NIST-compatible mass spectra;
- `figure/**/*.png`: resolution figures when figure generation is enabled.

> **Note:** The current implementation is a local browser-based application,
> not a publicly hosted web server or a platform-independent executable.

## Python API

The main programmatic entry point is `data_resolution`:

```python
from DeepCPR import data_resolution

data_resolution(
    dataset_path="path/to/raw/files",
    DeepCS_path="path/to/DeepCS.h5",
    DeepCPR_path="path/to/DeepCPR.h5",
    save_path="path/to/results",
    generate_image=False,
)
```

## Example notebook

The complete DeepCPR resolution workflow and representative results are
demonstrated in [`example.ipynb`](https://github.com/YuChuanxiu/DeepCPR/blob/main/example.ipynb).
The notebook includes an automatic chromatographic resolution example and an
OPLS-DA analysis using a resolved human plasma peak table.

## Advanced usage

### Adaptive resolution for more than five co-eluting components

The original network predicts five chromatographic profiles per forward pass.
The adaptive extension repeatedly applies the same network to the positive
reconstruction residual, estimates spectra with ITTFA/NNLS, removes duplicate
profiles, and jointly refits all retained components. The number of components
is therefore data-dependent and can exceed five while the model input remains
limited to 128 retention-time scans.

```python
from DeepCPR import data_resolution

data_resolution(
    dataset_path="path/to/raw/files",
    DeepCS_path="path/to/DeepCS.h5",
    DeepCPR_path="path/to/DeepCPR.h5",
    save_path="path/to/results",
    generate_image=False,
    adaptive=True,
    adaptive_kwargs={
        "max_iterations": 8,
        "max_components": 32,
        "min_improvement": 0.005,
    },
)
```

The direct segment-level API is `DeepCPRAdaptive`. The existing
`data_resolution` behavior is unchanged when `adaptive=False` (the default).

### TensorFlow-independent deployment with ONNX

The ONNX models provide a framework-independent inference route. Install the
ONNX dependencies with:

```bash
pip install -r requirements-onnx.txt
```

Both model exports are required for the complete workflow:

- `DeepCS.onnx` performs chromatographic segmentation;
- `DeepCPR.onnx` predicts chromatographic profiles.

Pass explicit `.onnx` paths to use ONNX Runtime directly:

```python
from DeepCPR import data_resolution

data_resolution(
    dataset_path="path/to/raw/files",
    DeepCS_path="path/to/DeepCS.onnx",
    DeepCPR_path="path/to/DeepCPR.onnx",
    save_path="path/to/results",
    generate_image=False,
    adaptive=True,
)
```

The ONNX adapter preserves the tensor layouts used by the Keras exports. The
DeepCPR profile model accepts `(batch, 128, 1, 800)` and returns
`(batch, 128, 1, 5)`.

For backward compatibility, an `.h5` path uses TensorFlow when TensorFlow is
available. If TensorFlow is unavailable, the loader can use a same-stem `.onnx`
file beside the requested `.h5` file; explicit ONNX paths are recommended for
TensorFlow-free deployment.

## Downstream analysis

The resolved peak tables can be used for downstream statistical analysis and
compound identification. `workflow.py` provides a seamless workflow from raw
datasets to peak tables, resolved mass spectra, and potential biomarkers.
The `msp_to_csv.py` utility converts DeepCPR MSP files to CSV format for tools
such as FastEI.

## Maintainers

222301019@csu.edu.cn
