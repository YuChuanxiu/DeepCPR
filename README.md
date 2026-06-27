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

# Information of maintainers
- 222301019@csu.edu.cn
