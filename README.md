# DeepCPR
Deep learning-based Chromatographic Profile Resolution 

------
DeepCPR is an easy-use tool to resolve untargeted GC-MS data automatically based on chromatographic prediction. The users simply enter the data path to be analyzed, and DeepCPR will divide the raw GC-MS data into segments and predict the chromatographic profiles for each segment. Then, the corresponding mass spectra are obtained by iterative MCR methods. After all the segments are resolved, the peak table of the GC-MS data will be generated, including retention time, peak area, and SNR information. The constructed OPLS-DA model performs discriminant analysis on the peak table and screens potential markers.

<div align="center">
<img src="https://github.com/YuChuanxiu/DeepCPR/blob/main/images/workflow-4_AC2.png" width=800 height=1000 />
</div>

# Package required: 
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).
- [python3](https://www.python.org/)    
- [tensorflow](https://www.tensorflow.org) 

By using the [`requirements.txt`](https://github.com/YuChuanxiu/DeepCPR/blob/main/requirements/pip/requirements.txt) file, it will install all the required packages.

    git clone https://github.com/YuChuanxiu/DeepCPR.git
    cd DeepCPR
    pip install -r requirements/pip/requirements.txt

# Information of maintainers
- 222301019@csu.edu.cn
e
