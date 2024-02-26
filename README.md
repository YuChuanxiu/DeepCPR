# DeepCPR
Deep learning-based Chromatographic Profile Resolution 

------
DeepCPR is an easy-use tool to resolve untargeted GC-MS data automatically based on chromatographic prediction. The users simply enter the data path to be analyzed, and DeepCPR will divide the raw GC-MS data into segments and predict the chromatographic profiles for each segment. Then, the corresponding mass spectra are obtained by iterative MCR methods. After all the segments are resolved, the peak table of the GC-MS data will be generated, including retention time, peak area, explained variances, and SNR information. The constructed OPLS-DA model performs discriminant analysis on the peak table and screens potential biomarkers. workflow.py provides the whole seamless procession from raw dataset into potential biomarkers. Compounds identification is realized by [`FastEI`](https://github.com/Qiong-Yang/FastEI/tree/main). Mass spectrometry identification can be performed directly using FastEI software or codes. Since the input of FastEI is in csv format, a conversion method is provided in msp_to_csv.py. 

<div align="center">
<img src="https://github.com/YuChuanxiu/DeepCPR/blob/main/images/workflow.tif" width=785 height=939 />
</div>

# Package required: 
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).
- [python3](https://www.python.org/)    
- [tensorflow](https://www.tensorflow.org) 

By using the [`requirements.txt`](https://github.com/YuChuanxiu/DeepCPR/blob/main/requirements/requirements.txt) file, it will install all the required packages.

    git clone https://github.com/YuChuanxiu/DeepCPR.git
    cd DeepCPR
    pip install -r requirements/pip/requirements.txt

# Example
The DeepCPR resolution process and results are shown in [`example.ipynb`](https://github.com/YuChuanxiu/DeepCPR/blob/main/example.ipynb), where DeepCPR resolutions are demonstrated using the lowest concentration data from the fatty acid dataset. For the OPLS-DA process, the peak table data is from a human plasma dataset, which is resolved by DeepCPR.

# Information of maintainers
- 222301019@csu.edu.cn
