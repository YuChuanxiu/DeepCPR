# DeepCPR

Deep learning-based Chromatographic Profile Resolution

## Overview

DeepCPR is a tool for automatic resolution of complex GC-MS data based on
chromatographic profile prediction. It divides raw GC-MS data into segments,
predicts chromatographic profiles, and estimates the corresponding mass
spectra using iterative multivariate curve resolution (MCR) methods. The
workflow produces resolved peak tables and mass spectra for downstream
compound identification and statistical analysis.

The repository also includes the leakage controlled repeated OPLS-DA analysis
used for the clinical plasma peak table. Resolved spectra can be searched against the NIST library
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
to create a Python 3.10 environment and [pip](https://pypi.org/project/pip/)
to install the Python packages. The Python version is intentionally not listed
as a pip requirement because pip cannot create or manage the interpreter itself.
Use a separate environment for each runtime option below; do not install the
TensorFlow and ONNX requirement files into the same environment unless model
conversion is required.

Clone the repository and enter its root directory:

```bash
git clone https://github.com/YuChuanxiu/DeepCPR.git
cd DeepCPR
```

### H5/TensorFlow CPU-compatible inference

Use this option on a machine without a supported NVIDIA GPU or when CPU-only
execution is preferred. The requirements file installs the CPU-compatible
TensorFlow Python package:

```bash
conda create -n DeepCPR_tf210_cpu python=3.10
conda activate DeepCPR_tf210_cpu
python -m pip install -r requirements.txt
python -m pip check
```

Verify the installation:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

TensorFlow should report version `2.10.0`. An empty GPU list is expected in a
CPU-only environment.

### H5/TensorFlow GPU inference on Windows

TensorFlow 2.10 is the last TensorFlow release with native Windows GPU support.
There is no separate GPU-specific Python requirements file. GPU users should
install the same Python dependencies as the CPU-compatible environment, then
add the matching CUDA runtime and cuDNN libraries in the Conda environment:

```bash
conda create -n DeepCPR_tf210_gpu python=3.10
conda activate DeepCPR_tf210_gpu
conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
python -m pip install -r requirements.txt
python -m pip check
```

The GPU environment is optional. For small chromatographic files, GPU execution
can be slower than CPU execution because model and data transfer overhead may
dominate the inference time. Verify that TensorFlow can see the GPU:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

The output must contain at least one `PhysicalDevice` with `device_type='GPU'`.
If the list is empty, DeepCPR will still run but TensorFlow inference will use
the CPU. These native Windows GPU instructions were validated with TensorFlow
2.10, CUDA 11.2, cuDNN 8.1, and an NVIDIA GPU; newer native Windows TensorFlow
releases do not provide the same CUDA path.

### ONNX Runtime CPU inference

The current ONNX adapter explicitly uses `CPUExecutionProvider`; therefore the
published ONNX route is CPU-only and does not require TensorFlow, CUDA, or
cuDNN:

```bash
conda create -n DeepCPR_onnx python=3.10
conda activate DeepCPR_onnx
python -m pip install -r requirements-onnx.txt
python -m pip check
```

Verify the ONNX provider:

```bash
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

The expected runtime version is `1.13.1`, and the provider list must include
`CPUExecutionProvider`.

The existing `tf210_onnx` environment used during model conversion contains
both TensorFlow and ONNX conversion packages. Those extra packages are not
required to run the released `.onnx` models.

The runtime choices are summarized below:

| Runtime | Model format | Installation |
|---|---|---|
| TensorFlow CPU-compatible | `.h5` | `python -m pip install -r requirements.txt` |
| TensorFlow GPU (Windows, CUDA 11.2/cuDNN 8.1) | `.h5` | Install the same requirements after the CUDA/cuDNN step |
| ONNX Runtime CPU | `.onnx` | `python -m pip install -r requirements-onnx.txt` |

The TensorFlow environments reproduce the original H5-based workflow. The ONNX
environment supports inference without requiring TensorFlow and is portable to
other platforms that support ONNX Runtime.

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
The notebook includes an automatic chromatographic resolution example.

## Reproducing the clinical plasma OPLS-DA analysis

The corrected clinical analysis starts from the fixed 136 sample by 53
variable peak area matrix in [`example/PlasmaTable.csv`](example/PlasmaTable.csv).
The first 61 rows are healthy controls and are assigned `+1`. The following 75
rows are men with semen abnormalities and are assigned `-1`. These labels are
intentionally defined in [`DeepCPR/q2_nested_cv.py`](DeepCPR/q2_nested_cv.py),
and the script checks the matrix dimensions and sample order before analysis.

The analysis uses 100 repeated stratified outer ten fold divisions. Scaling
parameters are calculated from the outer calibration samples only. Within each
outer calibration sample, ten fold Q2 evaluation selects between zero and nine
orthogonal components. An additional component is retained when its incremental
Q2 is at least 0.01. Outer validation samples are used only for class prediction.
VIP4t is calculated in each outer calibration model and averaged over the 1000
models.

From the repository root, run:

```bash
python DeepCPR/q2_nested_cv.py --input example/PlasmaTable.csv --output-dir example/results --repeats 100 --seed 20260904 --max-components 9 --inner-folds 10 --q2-threshold 0.01
```

The OPLS-DA and VIP4t calculations are implemented in
[`DeepCPR/oplsda_core.py`](DeepCPR/oplsda_core.py). With the parameters above, the mean accuracy is 98.54 percent, the mean
sensitivity is 97.37 percent, and the mean specificity is 99.97 percent.

This analysis evaluates OPLS-DA performance conditional on the supplied fixed
peak area matrix. It does not repeat chromatographic peak extraction or
retention time matching and is not an independent clinical validation.

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

The ONNX models provide a framework-independent CPU inference route. Activate
the ONNX environment created above before running the workflow:

```bash
conda activate DeepCPR_onnx
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
compound identification. `workflow.py` processes raw datasets into peak tables
and resolved mass spectra. The clinical OPLS-DA scripts are
[`DeepCPR/q2_nested_cv.py`](DeepCPR/q2_nested_cv.py) and
[`DeepCPR/oplsda_core.py`](DeepCPR/oplsda_core.py).
The `msp_to_csv.py` utility converts DeepCPR MSP files to CSV format for tools
such as FastEI.

## Fine-tuning model
An additional Python program named fine_tune_deepcpr.py is provided for fine tuning with additional GC MS segments and corresponding chromatographic profile labels that satisfy the required model dimensions.

## Maintainers

222301019@csu.edu.cn
