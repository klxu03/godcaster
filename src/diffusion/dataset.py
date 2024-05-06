from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

from datasets import load_dataset
from config import get_weights_file_path, get_config
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, random_split

class AnimeImageTextDataset(Dataset):
    def __init__(self, ds, tokenizer, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds[index]

        text = item["tags"]

        text_input_tokens = self.tokenizer.encode(text).ids
        text_num_padding_tokens = self.seq_len - len(text_input_tokens)

        assert text_num_padding_tokens >= 0, "The source sentence is too long"

        text_input = torch.cat(
            [
                torch.tensor(text_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * text_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        text_mask = (text_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)

        return {
            "text_input": text_input,
            "text_mask": text_mask,
            "img": item["image"],
            "format": item["image"].format
        }

def get_or_build_tokenizer(config, dataset, vocab_size=30000):
    tokenizer_path = Path(config['tokenizer_file'])
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[PAD]"], vocab_size=vocab_size)

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["tags"]

    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/en/quicktour
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    dataset = load_dataset(config["datasource"], config["subset"], split="train")
    tokenizer = get_or_build_tokenizer(config, dataset)

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(dataset))
    val_ds_size = len(dataset) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(dataset, [train_ds_size, val_ds_size])

    train_ds = AnimeImageTextDataset(train_ds_raw, tokenizer, config["seq_len"])
    val_ds = AnimeImageTextDataset(val_ds_raw, tokenizer, config["seq_len"])

    return train_ds, val_ds, tokenizer

def main():
    config = get_config()
    get_ds(config)

if __name__ == "__main__":
    main()