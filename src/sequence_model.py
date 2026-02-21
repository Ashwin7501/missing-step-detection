"""
Sequence Model Module - Transformer and LSTM models for procedural flow.
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelOutput:
    loss: Optional[object] = None
    logits: Optional[object] = None
    embeddings: Optional[object] = None


if HAS_TORCH:
    class StepEncoder(nn.Module):
        """Encoder for procedural steps using BERT or simple embeddings."""

        def __init__(self, model_name="bert-base-uncased", hidden_size=768, dropout=0.1):
            super().__init__()
            self.hidden_size = hidden_size
            try:
                from transformers import AutoModel
                self.bert = AutoModel.from_pretrained(model_name)
                bert_hidden = self.bert.config.hidden_size
            except Exception:
                self.bert = None
                bert_hidden = 256
                self.embedding = nn.Embedding(30000, bert_hidden)
                self.lstm = nn.LSTM(bert_hidden, bert_hidden//2, bidirectional=True, batch_first=True)

            self.projection = nn.Linear(bert_hidden, hidden_size) if bert_hidden != hidden_size else nn.Identity()
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(hidden_size)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            if self.bert is not None:
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                pooled = outputs.last_hidden_state[:, 0, :]
            else:
                embedded = self.embedding(input_ids)
                lstm_out, _ = self.lstm(embedded)
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (lstm_out * mask).sum(1) / mask.sum(1).clamp(min=1)
            return self.layer_norm(self.dropout(self.projection(pooled)))


    class ProcedureSequenceModel(nn.Module):
        """BiLSTM for modeling procedural flow."""

        def __init__(self, input_size=768, hidden_size=256, num_layers=2, dropout=0.3, bidirectional=True):
            super().__init__()
            self.hidden_size = hidden_size
            self.bidirectional = bidirectional
            self.input_projection = nn.Linear(input_size, hidden_size)
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                               dropout=dropout if num_layers > 1 else 0,
                               bidirectional=bidirectional, batch_first=True)
            self.output_size = hidden_size * 2 if bidirectional else hidden_size
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(self.output_size)

        def forward(self, step_embeddings, lengths=None):
            projected = self.input_projection(step_embeddings)
            output, (hidden, _) = self.rnn(projected)
            if self.bidirectional:
                final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                final_hidden = hidden[-1]
            return self.layer_norm(self.dropout(output)), final_hidden


    class TransitionPredictor(nn.Module):
        """Predicts whether a transition between steps is valid."""

        def __init__(self, input_size=512, hidden_size=256, num_classes=2, dropout=0.3):
            super().__init__()
            self.combine = nn.Sequential(
                nn.Linear(input_size * 2, hidden_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(hidden_size, num_classes)
            self.confidence = nn.Sequential(
                nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
            )

        def forward(self, step1_emb, step2_emb):
            combined = torch.cat([step1_emb, step2_emb], dim=1)
            hidden = self.combine(combined)
            return self.classifier(hidden), self.confidence(hidden)


    class StepCoherenceModel(nn.Module):
        """Full model combining step encoder, sequence model, and predictors."""

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.step_encoder = StepEncoder(
                model_name=config.get("transformer_model", "bert-base-uncased"),
                hidden_size=config.get("hidden_size", 768),
            )
            self.sequence_model = ProcedureSequenceModel(
                input_size=config.get("hidden_size", 768),
                hidden_size=config.get("lstm_hidden_size", 256),
            )
            self.transition_predictor = TransitionPredictor(
                input_size=self.sequence_model.output_size,
                hidden_size=config.get("lstm_hidden_size", 256),
            )


    class TransitionDataset(Dataset):
        """Dataset for training transition predictor."""

        def __init__(self, positive_pairs, negative_pairs, tokenizer, max_length=128):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.samples = [(s1, s2, 1) for s1, s2 in positive_pairs] + [(s1, s2, 0) for s1, s2 in negative_pairs]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s1, s2, label = self.samples[idx]
            enc1 = self.tokenizer(s1, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            enc2 = self.tokenizer(s2, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            return {
                "input_ids_1": enc1["input_ids"].squeeze(0),
                "attention_mask_1": enc1["attention_mask"].squeeze(0),
                "input_ids_2": enc2["input_ids"].squeeze(0),
                "attention_mask_2": enc2["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            }


    class ModelTrainer:
        """Training utilities for sequence models."""

        def __init__(self, model, config, device="cuda" if torch.cuda.is_available() else "cpu"):
            self.model = model.to(device)
            self.config = config
            self.device = device
            self.optimizer = AdamW(model.parameters(), lr=config.get("learning_rate", 2e-5))
            self.criterion = nn.CrossEntropyLoss()
            self.best_val_loss = float("inf")

        def train_epoch(self, train_loader, epoch):
            self.model.train()
            total_loss, total_correct, total_samples = 0, 0, 0
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                emb1 = self.model.step_encoder(batch["input_ids_1"], batch["attention_mask_1"])
                emb2 = self.model.step_encoder(batch["input_ids_2"], batch["attention_mask_2"])
                logits, _ = self.model.transition_predictor(emb1, emb2)
                loss = self.criterion(logits, batch["label"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
                total_correct += (logits.argmax(dim=1) == batch["label"]).sum().item()
                total_samples += batch["label"].size(0)
            return {"train_loss": total_loss / len(train_loader), "train_accuracy": total_correct / total_samples}

        def evaluate(self, val_loader):
            self.model.eval()
            total_loss, total_correct, total_samples = 0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    emb1 = self.model.step_encoder(batch["input_ids_1"], batch["attention_mask_1"])
                    emb2 = self.model.step_encoder(batch["input_ids_2"], batch["attention_mask_2"])
                    logits, _ = self.model.transition_predictor(emb1, emb2)
                    loss = self.criterion(logits, batch["label"])
                    total_loss += loss.item()
                    total_correct += (logits.argmax(dim=1) == batch["label"]).sum().item()
                    total_samples += batch["label"].size(0)
            return {"val_loss": total_loss / len(val_loader), "val_accuracy": total_correct / total_samples}

        def train(self, train_loader, val_loader, num_epochs, save_dir=None):
            history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
            for epoch in range(num_epochs):
                train_metrics = self.train_epoch(train_loader, epoch)
                val_metrics = self.evaluate(val_loader)
                history["train_loss"].append(train_metrics["train_loss"])
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_accuracy"].append(val_metrics["val_accuracy"])
                logger.info(f"Epoch {epoch+1}: train_loss={train_metrics['train_loss']:.4f}, val_acc={val_metrics['val_accuracy']:.4f}")
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    if save_dir:
                        torch.save({"model_state_dict": self.model.state_dict(), "config": self.config},
                                   f"{save_dir}/best_model.pt")
            return history

else:
    # Stubs when PyTorch is not available
    class StepEncoder: pass
    class ProcedureSequenceModel: pass
    class TransitionPredictor: pass
    class StepCoherenceModel: pass
    class TransitionDataset: pass
    class ModelTrainer: pass
