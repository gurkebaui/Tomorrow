import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformer_block import TransformerBlock, get_data_loader


def train(
    num_epochs: int = 10,
    batch_size: int = 8,
    d_model: int = 512,
    num_heads: int = 8,
    dropout: float = 0.1,
    learning_rate: float = 3e-4,
    block_size: int = 256,
):
    """Train a simple transformer on wikitext data."""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = TransformerBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
    model = model.to(device)

    # Create data loader (streaming from wikitext-2)
    dataloader, tokenizer = get_data_loader(
        dataset_name="wikitext",
        split="train",
        batch_size=batch_size,
        block_size=block_size,
    )

    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    step = 0
    running_loss = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            output, _ = model(input_ids)  # (B, S, D)
            loss = criterion(
                output.view(-1, len(tokenizer)), labels.view(-1)
            )  # Cross-entropy on token predictions

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % 50 == 0:
                avg_loss = running_loss / 50
                print(f"Step {step}, Loss: {avg_loss:.4f}")
                running_loss = 0.0

    # Save model
    torch.save(model.state_dict(), "transformer_block.pth")
    print(f"\nTraining complete! Model saved to transformer_block.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple Transformer block")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--block-size", type=int, default=256, help="Sequence length (tokens)"
    )

    args = parser.parse_args()
    train(**vars(args))
