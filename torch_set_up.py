import torch

"""
torch set_up for the inference
"""
# setting device on GPU if available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    print('Using device:', DEVICE)
    print()

    # Additional Info when using cuda
    if DEVICE.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    print("we will use the following device : ", DEVICE)


