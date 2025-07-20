from pathlib import Path
import torch
import numpy as np

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MAX_LEN  = 501          # fixed global length (pad / truncate)
ALPHABET = "ACGU"       # order = channels

def read_fasta(path, label):
    seqs, labs = [], []
    with Path(path).open() as fh:
        for line in fh:
            if line.startswith(">"): continue
            s = line.strip().upper()
            seqs.append(s)
            labs.append(label)
    return seqs, labs

def encode_batch(seqs, max_len=None, alphabet=None, pad_char='N'):
    """
    Convert a list of RNA sequences (variable length) into a 4×L one-hot tensor batch.
    - Pads with 'N' (uniform 0.25 across channels) or truncates to max_len.
    - Unknown characters also become uniform 0.25.
    
    Returns: torch.FloatTensor of shape (B, 1, 4, L)
    """
    if max_len is None:
        max_len = MAX_LEN
    if alphabet is None:
        alphabet = ALPHABET
    
    B = len(seqs)
    L = max_len
    A = len(alphabet)
    
    # Start with uniform 0.25 everywhere (so padding & unknown are already correct)
    X = np.full((B, 1, A, L), 1.0 / A, dtype=np.float32)
    
    # Precompute mapping from base → index
    idx_map = {b:i for i,b in enumerate(alphabet)}
    
    for b, seq in enumerate(seqs):
        # Pad / truncate
        s = (seq + pad_char * (L - len(seq)))[:L]
        for i, base in enumerate(s):
            if base in idx_map:          # set one-hot
                X[b, 0, :, i] = 0.0
                X[b, 0, idx_map[base], i] = 1.0
            # else: leave the uniform 0.25 (already there)
    return torch.from_numpy(X)

def load_dataset(pos_fasta, neg_fasta):
    """
    Reads positive & negative FASTA files, encodes them in one call.

    Returns:
        X : (N,1,4,L) FloatTensor
        y : (N,)      FloatTensor (0/1)
    """
    pos_seqs, pos_labels = read_fasta(pos_fasta, 1)
    neg_seqs, neg_labels = read_fasta(neg_fasta, 0)
    
    all_seqs = pos_seqs + neg_seqs
    all_labels = np.array(pos_labels + neg_labels, dtype=np.float32)
    
    X = encode_batch(all_seqs)            # (N,1,4,L)
    y = torch.from_numpy(all_labels)      # (N,)
    return X, y


def random_rna(length=MAX_LEN, rng_seed=42, alphabet=ALPHABET):
    rng = np.random.default_rng(rng_seed)
    return ''.join(rng.choice(list(alphabet), size=length, replace=True))

def batch_random_rna(batch_size=1000, length=MAX_LEN, rng_seed=42, alphabet=ALPHABET):
    return [random_rna(length, rng_seed, alphabet) for _ in range(batch_size)]
