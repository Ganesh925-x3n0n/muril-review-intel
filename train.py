"""
train.py — Fine-tune MuRIL for dual review classification
  - Sentiment: positive / negative / neutral
  - Authenticity: genuine / fake

Usage:
  python train.py --data test_data.csv --output muril_review_model --epochs 5

Requirements:
  pip install transformers datasets torch scikit-learn pandas
"""

import argparse
import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ── Label maps ────────────────────────────────────────────────────────────────
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
FAKE_LABELS      = ["genuine", "fake"]


# ── Dataset ───────────────────────────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts, sentiment_labels, fake_labels, tokenizer, max_len=128):
        self.texts           = texts
        self.sentiment_labels = sentiment_labels
        self.fake_labels     = fake_labels
        self.tokenizer       = tokenizer
        self.max_len         = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sentiment":      torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
            "fake":           torch.tensor(self.fake_labels[idx],      dtype=torch.long),
        }


# ── Dual-head model ────────────────────────────────────────────────────────────
class MuRILDualClassifier(nn.Module):
    def __init__(self, model_name, num_sentiment, num_fake, dropout=0.3):
        super().__init__()
        self.encoder           = AutoModel.from_pretrained(model_name)
        hidden                 = self.encoder.config.hidden_size
        self.dropout           = nn.Dropout(dropout)
        self.sentiment_head    = nn.Linear(hidden, num_sentiment)
        self.fake_head         = nn.Linear(hidden, num_fake)

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])   # [CLS] token
        return self.sentiment_head(pooled), self.fake_head(pooled)


# ── Training loop ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    criterion  = nn.CrossEntropyLoss()

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sent_labels    = batch["sentiment"].to(device)
        fake_labels    = batch["fake"].to(device)

        optimizer.zero_grad()
        sent_logits, fake_logits = model(input_ids, attention_mask)

        loss = criterion(sent_logits, sent_labels) + criterion(fake_logits, fake_labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    model.eval()
    all_sent_preds, all_sent_true = [], []
    all_fake_preds, all_fake_true = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sent_logits, fake_logits = model(input_ids, attention_mask)

            all_sent_preds.extend(sent_logits.argmax(dim=1).cpu().numpy())
            all_sent_true.extend(batch["sentiment"].numpy())
            all_fake_preds.extend(fake_logits.argmax(dim=1).cpu().numpy())
            all_fake_true.extend(batch["fake"].numpy())

    return all_sent_preds, all_sent_true, all_fake_preds, all_fake_true


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="test_data.csv",        help="Path to CSV file")
    parser.add_argument("--output",  default="muril_review_model",   help="Output directory")
    parser.add_argument("--model",   default="google/muril-base-cased")
    parser.add_argument("--epochs",  type=int, default=5)
    parser.add_argument("--batch",   type=int, default=8)
    parser.add_argument("--lr",      type=float, default=2e-5)
    parser.add_argument("--maxlen",  type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} samples")

    # Encode labels
    sent_enc = LabelEncoder()
    fake_enc = LabelEncoder()
    sent_enc.classes_ = np.array(SENTIMENT_LABELS)
    fake_enc.classes_ = np.array(FAKE_LABELS)
    df["sent_id"] = sent_enc.transform(df["sentiment_label"])
    df["fake_id"] = fake_enc.transform(df["fake_label"])

    # Train/val split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment_label"])
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_ds = ReviewDataset(
        train_df["text"].tolist(), train_df["sent_id"].tolist(), train_df["fake_id"].tolist(),
        tokenizer, args.maxlen
    )
    val_ds = ReviewDataset(
        val_df["text"].tolist(), val_df["sent_id"].tolist(), val_df["fake_id"].tolist(),
        tokenizer, args.maxlen
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch)

    # Model
    print(f"Loading model: {args.model}")
    model = MuRILDualClassifier(
        args.model,
        num_sentiment=len(SENTIMENT_LABELS),
        num_fake=len(FAKE_LABELS)
    ).to(device)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    # Training
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        sent_preds, sent_true, fake_preds, fake_true = eval_epoch(model, val_loader, device)

        print(f"\nEpoch {epoch}/{args.epochs}  |  Train loss: {train_loss:.4f}")
        print("── Sentiment ──")
        print(classification_report(sent_true, sent_preds, target_names=SENTIMENT_LABELS))
        print("── Authenticity ──")
        print(classification_report(fake_true, fake_preds, target_names=FAKE_LABELS))

        # Save best model
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            os.makedirs(args.output, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output, "model.pt"))
            tokenizer.save_pretrained(args.output)
            # Save label maps
            with open(os.path.join(args.output, "label_map.json"), "w") as f:
                json.dump({
                    "sentiment_labels": SENTIMENT_LABELS,
                    "fake_labels":      FAKE_LABELS,
                    "base_model":       args.model,
                }, f, indent=2)
            print(f"  ✓ Model saved to {args.output}/")

    print("\nTraining complete!")
    print(f"Model saved at: {args.output}/")


if __name__ == "__main__":
    main()
