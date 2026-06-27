from DeepCPR.DeepCPR import data_resolution

# The input data path can be customized as a folder containing files in the 'CDF' format.
dataset_path = "example/data"

# Storage path can be customized
save_path = 'example2/'

# The model path can be customized according to the actual path.
DeepCS_path = 'example/DeepCS.h5'
DeepCPR_path = 'example/DeepCPR.h5'

data_resolution(dataset_path, DeepCS_path, DeepCPR_path, save_path, True)