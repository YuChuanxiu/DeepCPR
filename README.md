# DeepCPR
Deep learning-based Chromatographic Profile Resolution 

------
DeepCPR is an easy-use tool to resolve complex GC-MS data automatically based on chromatographic prediction. The users simply enter the data path to be analyzed, and DeepCPR will divide the raw GC-MS data into segments and predict the chromatographic profiles for each segment. Then, the corresponding mass spectra are obtained by iterative MCR methods. After all the segments are resolved, the peak table of the GC-MS data will be generated. The constructed OPLS-DA model performs discriminant analysis on the peak table and screens potential biomarkers of GC-MS data. workflow.py provides the whole seamless processes from raw dataset into peak table, resolved mass spectra, and potential biomarkers. Compounds identification can be realized by [`FastEI`](https://github.com/Qiong-Yang/FastEI/tree/main). FastEI is utilized as an alternative search tool for compounds that cannot be identified by matching the NIST library. Since the input of FastEI is in csv format, the resolved mass soectra by DeepCPR are in msp format, a conversion method is provided in msp_to_csv.py. 

<div align="center">
<img src="https://github.com/YuChuanxiu/DeepCPR/blob/main/workflow.png" width=785 height=913 />
</div>

# Package required
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).
- [python3](https://www.python.org/)    
- [tensorflow](https://www.tensorflow.org) 

By using the [`requirements.txt`](https://github.com/YuChuanxiu/DeepCPR/blob/main/requirements.txt) file, it will install all the required packages.

    git clone https://github.com/YuChuanxiu/DeepCPR.git
    cd DeepCPR
    pip install -r requirements.txt

# Example
The DeepCPR resolution workflow and representative results are demonstrated in [`example.ipynb`](https://github.com/YuChuanxiu/DeepCPR/blob/main/example.ipynb).

Download the pretrained models and example datasets from [Release v1.1.0](https://github.com/YuChuanxiu/DeepCPR/releases/tag/v1.1.0):

- `DeepCPR.h5`
- `DeepCS.h5`
- `data.zip`

Place `DeepCPR.h5` and `DeepCS.h5` in the `example` folder. Extract `data.zip` into the same folder so that the datasets are located at:

```text
example/data/

The resulting directory structure should be:

DeepCPR/
├── example/
│   ├── data/
│   ├── DeepCPR.h5
│   └── DeepCS.h5
└── example.ipynb
```

In the first example, the lowest-concentration samples from the fatty acid dataset are used to demonstrate the automatic chromatographic resolution workflow. The OPLS-DA workflow is demonstrated using a peak table extracted from a human plasma dataset resolved by DeepCPR.

# Local graphical user interface

To make DeepCPR accessible to users without programming experience, we provide
a Gradio-based local graphical interface in `DeepCPR/app.py`. The interface runs
in a web browser, while all data processing and model inference are performed
locally. Raw GC-MS data are not uploaded to an external server.

## Main functions

- Upload multiple GC-MS files in CDF or NetCDF format.
- Use DeepCS and DeepCPR models in H5 or ONNX format.
- Enable adaptive resolution for segments containing more than five co-eluting components.
- Optionally generate chromatographic resolution figures.
- Monitor the processing status and runtime.
- Preview peak tables, segment information, and resolution figures.
- Download all results as a ZIP archive.

The exported results include:

- `peak_area_table.csv`: merged peak table;
- `single/*.csv`: peak tables for individual files;
- `seg/*.csv`: segment information;
- `ms/**/*.msp`: NIST-compatible mass spectra;
- `figure/**/*.png`: resolution figures when figure generation is enabled.

## Installation and launch

First install the required dependencies:

```powershell
pip install -r requirements.txt
```

or
```powershell
pip install -r requirements-onnx.txt
```

Place the pretrained DeepCS and DeepCPR model files in the example directory, as described above. 
From the repository root, launch the interface in terminal with:
```powershell
python DeepCPR/app.py
```

The interface will open automatically at: http://127.0.0.1:7860.
Users can upload GC-MS data, configure the resolution options, start the analysis, inspect the results, and download the complete output archive.
***Note: This is a local browser-based application rather than a publicly
hosted web server. All uploaded data remain on the user's computer.***

# Adaptive resolution for more than five co-eluting components

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

# TensorFlow-independent deployment with ONNX

The trained DeepCPR models are also provided in ONNX format. This makes the
application usable for inference without installing TensorFlow: ONNX is a
framework-independent model format and ONNX Runtime executes the exported
models in a standard Python environment. For example, the same inference
environment may also contain PyTorch or another deep-learning framework; the
ONNX inference path does not depend on TensorFlow or PyTorch.

The TensorFlow/H5 files remain available for reproducing the original training
and inference setup. To use that route, install:

```powershell
pip install -r requirements.txt
```

For deployment without TensorFlow, install the ONNX inference dependencies:

```powershell
pip install -r requirements-onnx.txt
```

Both model exports are required for the complete workflow:

- `DeepCS.onnx` performs chromatographic segmentation.
- `DeepCPR.onnx` predicts chromatographic profiles.

Passing explicit `.onnx` paths selects ONNX Runtime directly. No TensorFlow
installation is needed in this mode. The existing Python API is unchanged:

Example with explicit ONNX paths:

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

The ONNX adapter uses `onnxruntime` and preserves the model tensor layouts used
by the Keras exports. The DeepCPR profile model accepts
`(batch, 128, 1, 800)` and returns `(batch, 128, 1, 5)`.

For backward compatibility, an `.h5` path uses TensorFlow when TensorFlow is
available. If TensorFlow is unavailable, the loader can use a same-stem `.onnx`
file beside the requested `.h5` file; explicit ONNX paths are recommended for
TensorFlow-free deployment.

# Information of maintainers
- 222301019@csu.edu.cn
