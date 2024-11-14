import torch
import tiktoken
import matplotlib.pyplot as plt
import os

from dataloader import create_dataloader_v1
from attention import MultiHeadAttention
from gpt import GPTModel,generate_text_simple
from gpt_download import download_and_load_gpt2


if __name__ == '__main__':  
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

