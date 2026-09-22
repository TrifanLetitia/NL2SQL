import json
from collections import Counter
import torch
from torch.utils.data import Dataset

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    return text.strip().split()


def build_vocab(samples, field: str, min_freq: int = 1):
    counter = Counter()
    for item in samples:
        counter.update(tokenize(item[field]))

    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)

    idx2word = {idx: word for word, idx in vocab.items()}
    return vocab, idx2word


def numericalize(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    unk_idx = vocab["<unk>"]
    return [vocab.get(tok, unk_idx) for tok in tokens]


class TextToSQLDataset(Dataset):
    def __init__(self, samples, src_vocab, trg_vocab):
        self.samples = samples
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        src_tokens = tokenize(item["question"])
        trg_tokens = ["<sos>"] + tokenize(item["sql"]) + ["<eos>"]

        src_ids = numericalize(src_tokens, self.src_vocab)
        trg_ids = numericalize(trg_tokens, self.trg_vocab)

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)


def pad_sequence(seq: torch.Tensor, max_len: int, pad_idx: int) -> torch.Tensor:
    if len(seq) < max_len:
        padding = torch.full((max_len - len(seq),), pad_idx, dtype=torch.long)
        return torch.cat([seq, padding], dim=0)
    return seq[:max_len]


def collate_fn(batch, src_pad_idx: int, trg_pad_idx: int, max_len: int = 50):
    src_batch, trg_batch = zip(*batch)
    src_batch = torch.stack([pad_sequence(x, max_len, src_pad_idx) for x in src_batch])
    trg_batch = torch.stack([pad_sequence(x, max_len, trg_pad_idx) for x in trg_batch])
    return src_batch, trg_batch

def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    idx2word = {idx: word for word, idx in vocab.items()}
    return vocab, idx2word