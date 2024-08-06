import torch
from torchsummary import summary
from model import Model

model_path = '/home/bagus/data/MuSe2024/perception_models/perception/aggressive/ds/model_101.pth'
loaded_model = torch.load(model_path, map_location=torch.device('cpu'))
print(loaded_model)

# for key, value in loaded_model.items():
#     print(f"Layer: {key}, Shape: {value.shape}")
# print(loaded_model)

# print model summary
# summary(loaded_model,(4096,256))