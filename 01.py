import torch123

torch_version = torch.__version__
cuda_version_pt = torch.version.cuda
cudnn_version_pt = torch.backends.cudnn.version()

print(f'PyTorch_version', torch_version)
print(f'CUDA_Version', cuda_version_pt)
print(f'CUDA_Available', torch.cuda.is_available())
print(f'cuDNN_Version', cudnn_version_pt)
print(f'GPU_Name', torch.cuda.get_device_name(0))