# -*- encoding: utf-8 -*-
'''

@File    :   main.py
@Time    :   2025/08/19 15:12:16
@Author  :   Yangshuo He
@Contact :   sugarhe58@gmail.com
'''

import torch
from pbb.utils import runexp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA: {torch.cuda.is_available()}")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else DEVICE)
print("MPS: ", torch.backends.mps.is_available())
print(f"Using device: {DEVICE}")

