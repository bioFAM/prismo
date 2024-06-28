import torch


def get_free_gpu_idx():
    """Get the index of the GPU with current lowest memory usage."""

    max_free_idx = 0
    max_free_mem = torch.cuda.mem_get_info(0)[0]
    for i in range(torch.cuda.device_count()):
        if torch.cuda.mem_get_info(i)[0] > max_free_mem:
            max_free_idx = i
            max_free_mem = torch.cuda.mem_get_info(i)[0]

    return max_free_idx
