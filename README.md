# DeepCPR
Deep learning-based Chromatographic Profile Resolution 

------
DeepCPR is an easy-use tool to resolve untargeted GC-MS data automatically based on chromatographic prediction. The users simply enter the data path to be analyzed, and DeepCPR will divide the raw GC-MS data into segments and predict the chromatographic profiles for each segment. Then, the corresponding mass spectra are obtained by iterative MCR methods. After all the segments are resolved, the peak table of the GC-MS data will be generated. The constructed OPLS-DA model performs discriminant analysis on the peak table and screens potential biomarkers of GC-MS data. workflow.py provides the whole seamless processes from raw dataset into peak table, resolved mass spectra, and potential biomarkers. Compounds identification is realized by [`FastEI`](https://github.com/Qiong-Yang/FastEI/tree/main). FastEI is utilized as an alternative search tool for compounds that cannot be identified by matching the NIST library. Since the input of FastEI is in csv format, the resolved mass soectra by DeepCPR are in msp format, a conversion method is provided in msp_to_csv.py. 

<div align="center">
<img src="https://github.com/YuChuanxiu/DeepCPR/blob/main/workflow.png" width=785 height=913 />
</div>

# Package required: 
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).
- [python3](https://www.python.org/)    
- [tensorflow](https://www.tensorflow.org) 

By using the [`requirements.txt`](https://github.com/YuChuanxiu/DeepCPR/blob/main/requirements.txt) file, it will install all the required packages.

    git clone https://github.com/YuChuanxiu/DeepCPR.git
    cd DeepCPR
    pip install -r requirements.txt

# Example
The DeepCPR resolution process and results are shown in [`example.ipynb`](https://github.com/YuChuanxiu/DeepCPR/blob/main/example.ipynb). First, the lowest concentration data from the fatty acid dataset is utilized to demonstrate the automatic resolution process. Models of DeepCPR and DeepCPR and datasets of fatty acid and amino acid are upload in release v1.1. In this example, models are downloaded and put into "example" folder. To show the OPLS-DA process, the peak table data is extracted from a human plasma dataset, which is resolved by DeepCPR.

# Information of maintainers
- 222301019@csu.edu.cn
