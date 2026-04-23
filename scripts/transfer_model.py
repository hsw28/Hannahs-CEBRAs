import torch
from cebra import CEBRA

# Instantiating the model with device management
model = CEBRA(device='CPU')  # This will automatically use CUDA if available

# Load the model
model_path = '/Users/Hannah/Programming/data_eyeblink/rat0307/cebra_variables/models/modelA1_dim2_0_2024-07-31_17-17-56_div2.pt'
model = CEBRA.load(model_path)

# If CEBRA supports device handling, ensure it's moved to CPU
# This step assumes CEBRA has internal methods to handle devices
# If CEBRA does not support it, you might need to skip this or handle it as per CEBRA's documentation
model.to('cpu')

# Save using CEBRA's method
model.save('/Users/Hannah/Programming/data_eyeblink/model_cpu.pth')
