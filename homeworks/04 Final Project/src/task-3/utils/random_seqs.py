import numpy as np
import torch

from utils.preprocess_fa import ALPHABET, MAX_LEN, DEVICE, SEED

rng = np.random.default_rng(SEED)

# -------- Helpers --------
def random_rna(length):
    return ''.join(rng.choice(list(ALPHABET), size=length, replace=True))

def encode_batch(seqs):
    B = len(seqs)
    X = np.zeros((B, 1, 4, MAX_LEN), dtype=np.float32)
    for b,s in enumerate(seqs):
        for i,c in enumerate(s):
            if c in ALPHABET:
                X[b,0,ALPHABET.index(c),i] = 1.0
            else:
                X[b,0,:,i] = 0.25
    return torch.from_numpy(X)
    
def get_random_seqs(n):
    seqs = [random_rna(MAX_LEN) for _ in range(n)]
    return seqs, encode_batch(seqs).to(DEVICE)