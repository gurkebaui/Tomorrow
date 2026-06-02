from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for text data using Hugging Face's streaming mode."""

    def __init__(
        self,
        dataset_name: str = "wikitext",
        split: str = "train",
        tokenizer=None,
        block_size: int = 256,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        # Stream the dataset without downloading everything at once
        ds = load_dataset(self.dataset_name, "wikitext-2-v1", streaming=True)[
            self.split
        ]

        for example in ds:
            text = example["text"]
            if not text.strip():
                continue

            # Tokenize and create chunks of block_size tokens
            inputs = self.tokenizer(
                text,
                truncation=False,  # Don't truncate; we'll chunk manually
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"][0]

            if len(input_ids) < self.block_size:
                continue

            # Split into chunks of block_size tokens
            for i in range(0, len(input_ids) - self.block_size + 1, self.block_size):
                chunk = input_ids[i : i + self.block_size]
                yield {
                    "input_ids": chunk[:-1],  # Input: all but last token
                    "labels": chunk[1:],  # Target: shift by one (next-token prediction)
                }


def get_data_loader(
    dataset_name: str = "wikitext",
    split: str = "train",
    batch_size: int = 8,
    block_size: int = 256,
    num_workers: int = 0,
):
    """Create a DataLoader with streaming text data."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Use GPT-2's pad token as eos for consistency
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = StreamingTextDataset(
        dataset_name=dataset_name,
        split=split,
        tokenizer=tokenizer,
        block_size=block_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda x: {
            "input_ids": torch.stack([item["input_ids"] for item in x]),
            "labels": torch.stack([item["labels"] for item in x]),
        },
    )

    return dataloader, tokenizer
