import gc
import torch
with torch.cuda.device('cuda:0'):
    torch.cuda.empty_cache()
gc.collect()