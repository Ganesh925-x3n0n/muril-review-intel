"""
absa_train.py — Train a dedicated MuRIL model for Aspect-Based Sentiment Analysis

What this file does:
  Step 1 — Auto-labels your existing CSV with aspect + aspect-sentiment columns
            using weak supervision (lexicon-based labeling)
  Step 2 — Fine-tunes a separate MuRIL model on those aspect labels
  Step 3 — Saves the model to muril_absa_model/

This is completely separate from train.py and does NOT touch your original model.

Aspects detected:
  product_quality, delivery, customer_support, value_for_money

Per-aspect sentiment labels:
  positive, negative, neutral

Usage:
  # Full pipeline (auto-label + train):
  python absa_train.py --data final_train.csv --output muril_absa_model

  # Only auto-label (inspect before training):
  python absa_train.py --data final_train.csv --label-only --labeled-output labeled_absa.csv

  # Train on an already-labeled CSV:
  python absa_train.py --data labeled_absa.csv --already-labeled --output muril_absa_model

Requirements:
  pip install transformers torch scikit-learn pandas numpy
"""

import argparse
import os
import json
import ast

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ── Constants ──────────────────────────────────────────────────────────────────
ASPECTS = ["product_quality", "delivery", "customer_support", "value_for_money"]
ASPECT_SENTIMENTS = ["negative", "neutral", "positive"]   # keep alphabetical for index stability
BASE_MODEL = "google/muril-base-cased"

# Label for "aspect not mentioned in this review"
NOT_MENTIONED = "none"
ASPECT_SENT_LABELS = ASPECT_SENTIMENTS + [NOT_MENTIONED]   # 4 classes per aspect head


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — WEAK SUPERVISION: Auto-label aspects from text
# ══════════════════════════════════════════════════════════════════════════════

# Multilingual keyword lexicon — same logic as absa.py but extended
ASPECT_LEXICON = {
    "product_quality": {
        "en": [
            "quality", "build", "material", "durable", "sturdy", "flimsy",
            "broken", "damaged", "scratched", "defective", "original", "fake",
            "genuine", "works", "stopped working", "performance", "battery",
            "solid", "poor quality", "good quality", "not working", "dead",
            "product", "item", "very poor", "worst", "excellent product",
        ],
        "ta": [
            "தரம்", "பொருள்", "உடைந்த", "நல்லா", "மோசம்", "வேலை செய்யல",
            "வேலை செய்யவில்லை", "கெட்டுவிட்டது", "original", "duplicate",
            "quality", "build", "scratched", "உடைந்து", "item", "product",
        ],
        "hi": [
            "क्वालिटी", "बिल्ड", "टूट", "खराब", "अच्छा", "बढ़िया",
            "नकली", "असली", "काम नहीं", "बेकार", "मटेरियल", "original",
            "toot", "kharab", "quality", "build", "product", "item",
        ],
    },
    "delivery": {
        "en": [
            "delivery", "shipping", "shipped", "package", "packaging",
            "arrived", "arrival", "on time", "late", "delayed", "fast",
            "quick", "slow", "courier", "dispatch", "box", "condition",
            "received", "transit", "packing", "packed",
        ],
        "ta": [
            "டெலிவரி", "delivery", "பேக்கிங்", "packing", "packaging",
            "வந்தது", "சீக்கிரம்", "late", "பார்சல்", "parcel",
            "item safe", "vandhuchu", "packing",
        ],
        "hi": [
            "डिलीवरी", "delivery", "पैकेजिंग", "packaging", "पैकिंग",
            "आया", "देर", "जल्दी", "courier", "shipment", "parcel",
            "samay pe", "time pe", "packing",
        ],
    },
    "customer_support": {
        "en": [
            "support", "service", "customer care", "customer support",
            "return", "refund", "replacement", "response", "helpful",
            "useless", "rude", "polite", "contact", "complaint", "resolved",
            "exchange", "helpline", "seller",
        ],
        "ta": [
            "customer care", "support", "service", "return", "refund",
            "replacement", "பதில்", "உதவி", "சேவை", "seller",
        ],
        "hi": [
            "customer care", "support", "सर्विस", "रिटर्न", "वापसी",
            "रिफंड", "बदलाव", "मदद", "जवाब", "उत्तर",
            "service", "return", "refund", "seller",
        ],
    },
    "value_for_money": {
        "en": [
            "price", "worth", "value", "expensive", "cheap", "affordable",
            "overpriced", "money", "rupee", "rupees", "budget", "cost",
            "paise", "investment", "waste of money", "totally worth",
            "paise vasool", "paisa vasool", "worth it", "not worth",
        ],
        "ta": [
            "விலை", "பணம்", "price", "worth", "value", "cheap", "costly",
            "பைசா", "rupee", "money", "paisa vasool", "paise", "worth",
        ],
        "hi": [
            "कीमत", "दाम", "पैसे", "पैसा", "रुपए", "महंगा", "सस्ता",
            "वैल्यू", "worth", "price", "budget", "paisa", "rupee",
            "paise vasool", "पैसा वसूल", "worth it",
        ],
    },
}

POSITIVE_SIGNALS = [
    # English
    "good", "great", "excellent", "amazing", "awesome", "best", "love",
    "perfect", "fast", "quick", "on time", "helpful", "worth", "value",
    "satisfied", "happy", "recommend", "solid", "works", "neat", "clean",
    "original", "genuine", "totally worth", "paise vasool", "worth it",
    "superb", "fantastic", "wonderful", "smooth", "prompt",
    # Tamil
    "நல்லா", "நன்றி", "வாங்கலாம்", "safe", "சீக்கிரம்", "super",
    "vera level", "kandippa",
    # Hindi
    "अच्छा", "बढ़िया", "खुश", "पैसा वसूल", "सही", "बेहतरीन",
    "maza", "accha", "badhiya", "sahi", "khush",
]

NEGATIVE_SIGNALS = [
    # English
    "bad", "poor", "terrible", "awful", "worst", "hate", "broken", "damaged",
    "scratched", "defective", "fake", "slow", "late", "delayed", "useless",
    "rude", "unhelpful", "expensive", "overpriced", "waste", "stopped working",
    "not working", "flimsy", "return", "refund", "dead", "cheap quality",
    "very poor", "poor quality", "do not buy", "don't buy", "disappointed",
    "pathetic", "horrible", "never buy", "stop working", "not worth",
    # Tamil
    "மோசம்", "உடைந்து", "வேலை செய்யவில்லை", "வேண்டாம்", "கெட்டு",
    "oru vaaram kooda", "work aagala",
    # Hindi
    "खराब", "बेकार", "टूट", "नकली", "बर्बाद", "गंदा", "वापस",
    "kharab", "bekar", "toot", "nakli", "dobara mat lena",
]


def detect_aspect(text: str, aspect: str) -> bool:
    """Return True if the aspect is mentioned in the text."""
    text_lower = text.lower()
    for keywords in ASPECT_LEXICON[aspect].values():
        if any(kw.lower() in text_lower for kw in keywords):
            return True
    return False


def score_polarity(text: str, aspect: str) -> str:
    """Return positive/negative/neutral for a given aspect in text."""
    text_lower = text.lower()
    words = text_lower.split()

    # Find positions of aspect keywords
    aspect_positions = []
    for keywords in ASPECT_LEXICON[aspect].values():
        for kw in keywords:
            if kw.lower() in text_lower:
                for i, w in enumerate(words):
                    if kw.lower() in w or w in kw.lower():
                        aspect_positions.append(i)

    # Context window ±6 words around aspect mention
    if aspect_positions:
        context = []
        for pos in aspect_positions:
            start = max(0, pos - 6)
            end   = min(len(words), pos + 7)
            context.extend(words[start:end])
        context_str = " ".join(context)
    else:
        context_str = text_lower

    pos_count = sum(1 for sig in POSITIVE_SIGNALS if sig.lower() in context_str)
    neg_count = sum(1 for sig in NEGATIVE_SIGNALS if sig.lower() in context_str)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


def auto_label_row(text: str) -> dict:
    """
    Auto-label a single review with aspect presence + per-aspect sentiment.
    Returns a dict of {aspect: sentiment_or_none}
    """
    labels = {}
    for aspect in ASPECTS:
        if detect_aspect(text, aspect):
            labels[aspect] = score_polarity(text, aspect)
        else:
            labels[aspect] = NOT_MENTIONED
    return labels


def auto_label_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Add aspect label columns to the dataframe."""
    print("Auto-labeling aspects using weak supervision...")
    aspect_labels = df["text"].apply(auto_label_row)

    for aspect in ASPECTS:
        df[aspect] = aspect_labels.apply(lambda x: x[aspect])

    # Stats
    print("\nAspect coverage in dataset:")
    for aspect in ASPECTS:
        mentioned = (df[aspect] != NOT_MENTIONED).sum()
        print(f"  {aspect:<25}: {mentioned:>4} / {len(df)} reviews ({mentioned/len(df)*100:.1f}%)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATASET & MODEL
# ══════════════════════════════════════════════════════════════════════════════

class ABSADataset(Dataset):
    """
    Each sample produces:
      - input_ids, attention_mask  (from MuRIL tokenizer)
      - one label per aspect       (0=negative, 1=neutral, 2=positive, 3=none)
    """
    def __init__(self, texts, aspect_label_matrix, tokenizer, max_len=128):
        self.texts               = texts
        self.aspect_label_matrix = aspect_label_matrix   # shape: (N, num_aspects)
        self.tokenizer           = tokenizer
        self.max_len             = max_len

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
            "aspect_labels":  torch.tensor(
                self.aspect_label_matrix[idx], dtype=torch.long
            ),
        }


class MuRILABSAModel(nn.Module):
    """
    MuRIL encoder + one classification head per aspect.
    Each head classifies: negative / neutral / positive / none (not mentioned)

    Architecture:
      MuRIL [CLS]
          │
          ├── product_quality_head  → 4 classes
          ├── delivery_head         → 4 classes
          ├── customer_support_head → 4 classes
          └── value_for_money_head  → 4 classes
    """
    def __init__(self, model_name, num_aspects, num_classes_per_aspect, dropout=0.3):
        super().__init__()
        self.encoder  = AutoModel.from_pretrained(model_name)
        hidden        = self.encoder.config.hidden_size
        self.dropout  = nn.Dropout(dropout)
        self.num_aspects = num_aspects

        # One linear head per aspect
        self.aspect_heads = nn.ModuleList([
            nn.Linear(hidden, num_classes_per_aspect)
            for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])   # [CLS] token

        # Each head produces logits for its aspect
        logits = [head(pooled) for head in self.aspect_heads]
        return logits   # list of (batch, num_classes) tensors


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def encode_aspect_labels(df: pd.DataFrame) -> np.ndarray:
    """Convert string aspect labels to integer indices."""
    label2idx = {label: i for i, label in enumerate(ASPECT_SENT_LABELS)}
    matrix = np.zeros((len(df), len(ASPECTS)), dtype=int)
    for col_idx, aspect in enumerate(ASPECTS):
        matrix[:, col_idx] = df[aspect].map(label2idx).fillna(
            label2idx[NOT_MENTIONED]
        ).astype(int)
    return matrix


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    criterion  = nn.CrossEntropyLoss()
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        aspect_labels  = batch["aspect_labels"].to(device)   # (batch, num_aspects)

        optimizer.zero_grad()
        logits_list = model(input_ids, attention_mask)

        # Sum loss across all aspect heads
        loss = sum(
            criterion(logits_list[i], aspect_labels[:, i])
            for i in range(len(ASPECTS))
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    model.eval()
    # Collect preds and true labels per aspect
    all_preds = [[] for _ in ASPECTS]
    all_true  = [[] for _ in ASPECTS]

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aspect_labels  = batch["aspect_labels"]

            logits_list = model(input_ids, attention_mask)
            for i in range(len(ASPECTS)):
                preds = logits_list[i].argmax(dim=1).cpu().numpy()
                true  = aspect_labels[:, i].numpy()
                all_preds[i].extend(preds)
                all_true[i].extend(true)

    return all_preds, all_true


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train MuRIL ABSA model — aspect detection + sentiment"
    )
    parser.add_argument("--data",           default="final_train.csv",
                        help="Input CSV (must have 'text' column)")
    parser.add_argument("--output",         default="muril_absa_model",
                        help="Directory to save trained ABSA model")
    parser.add_argument("--model",          default=BASE_MODEL,
                        help="HuggingFace model name")
    parser.add_argument("--epochs",         type=int,   default=5)
    parser.add_argument("--batch",          type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--maxlen",         type=int,   default=128)
    parser.add_argument("--label-only",     action="store_true",
                        help="Only auto-label the CSV, do not train")
    parser.add_argument("--labeled-output", default="labeled_absa.csv",
                        help="Where to save the auto-labeled CSV")
    parser.add_argument("--already-labeled", action="store_true",
                        help="Skip auto-labeling, CSV already has aspect columns")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  {len(df)} samples loaded.")

    # ── Auto-label if needed ───────────────────────────────────────────────────
    if not args.already_labeled:
        df = auto_label_dataset(df)
        df.to_csv(args.labeled_output, index=False)
        print(f"\nAuto-labeled CSV saved to: {args.labeled_output}")
        print("  Tip: Open this file and manually correct obvious errors")
        print("       before rerunning with --already-labeled for better results.\n")

        if args.label_only:
            print("--label-only flag set. Stopping here.")
            return
    else:
        # Validate aspect columns exist
        missing = [a for a in ASPECTS if a not in df.columns]
        if missing:
            print(f"Error: Missing aspect columns: {missing}")
            print(f"Run without --already-labeled to auto-generate them.")
            return

    # ── Encode labels ──────────────────────────────────────────────────────────
    label_matrix = encode_aspect_labels(df)
    texts        = df["text"].tolist()

    print(f"\nLabel distribution per aspect:")
    for i, aspect in enumerate(ASPECTS):
        unique, counts = np.unique(label_matrix[:, i], return_counts=True)
        dist = {ASPECT_SENT_LABELS[u]: c for u, c in zip(unique, counts)}
        print(f"  {aspect:<25}: {dist}")

    # ── Train / val split ──────────────────────────────────────────────────────
    indices          = list(range(len(texts)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_texts  = [texts[i] for i in train_idx]
    val_texts    = [texts[i] for i in val_idx]
    train_labels = label_matrix[train_idx]
    val_labels   = label_matrix[val_idx]

    print(f"\nTrain: {len(train_texts)} | Val: {len(val_texts)}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_ds = ABSADataset(train_texts, train_labels, tokenizer, args.maxlen)
    val_ds   = ABSADataset(val_texts,   val_labels,   tokenizer, args.maxlen)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch)

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model = MuRILABSAModel(
        model_name=args.model,
        num_aspects=len(ASPECTS),
        num_classes_per_aspect=len(ASPECT_SENT_LABELS),
        dropout=0.3,
    ).to(device)

    # ── Optimizer + scheduler ──────────────────────────────────────────────────
    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_loss = float("inf")
    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        all_preds, all_true = eval_epoch(model, val_loader, device)

        print(f"Epoch {epoch}/{args.epochs}  |  Train Loss: {train_loss:.4f}")

        for i, aspect in enumerate(ASPECTS):
            print(f"\n  ── {aspect.replace('_', ' ').title()} ──")
            print(classification_report(
                all_true[i], all_preds[i],
                target_names=ASPECT_SENT_LABELS,
                zero_division=0,
            ))

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            os.makedirs(args.output, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output, "absa_model.pt"))
            tokenizer.save_pretrained(args.output)

            # Save metadata
            meta = {
                "base_model":           args.model,
                "aspects":              ASPECTS,
                "aspect_sent_labels":   ASPECT_SENT_LABELS,
                "not_mentioned_label":  NOT_MENTIONED,
            }
            with open(os.path.join(args.output, "absa_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            print(f"  ✓ Best model saved to {args.output}/")

    print(f"\nTraining complete!")
    print(f"Model saved at : {args.output}/absa_model.pt")
    print(f"Metadata saved : {args.output}/absa_meta.json")
    print(f"\nNext step: run absa_predict.py to use this model.")


if __name__ == "__main__":
    main()
