#!/usr/bin/env python
# coding: utf-8

import warnings as w
w.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# device, tasks, and data paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
incremental_tasks = [5, 20, 25, 30]
base_path = "data/"
# os.getenv("DATA_PATH", "data")
DATASETS = {
    "BBBP":  os.path.join(base_path, "BBBP.csv"),
    "NP":    os.path.join(base_path, "NP.csv"),
    "Toxcast": os.path.join(base_path, "Toxcast.csv"),
    "Sider":   os.path.join(base_path, "Sider.csv"),
    "Sweet":   os.path.join(base_path, "explsweet.csv"),
    "Bitter":  os.path.join(base_path, "explbitter.csv"),
    "Tox21":   os.path.join(base_path, "Tox21.csv"),
    "ClinTox": os.path.join(base_path, "clintox.csv"),
}

# ─── MODEL / DATA CLASSES ─────────────────────────────────────────────────────

class MoLFormerFeatureExtractor(nn.Module):
    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.molformer = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.molformer.eval()
        for p in self.molformer.parameters(): p.requires_grad = False

    def forward(self, smiles_list: list[str]):
        tokens = self.tokenizer(
            smiles_list, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            return self.molformer(**tokens).last_hidden_state[:, 0, :]

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class SMILESDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.smiles = df["SMILES"].tolist()
        self.labels = df["Label"].to_numpy()

    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.smiles[i], self.labels[i]

class EWC:
    def __init__(self, model: nn.Module, dataloader: DataLoader):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.fisher = {}
        self.optimal_params = {}

    def compute_fisher_information(self):
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()
        num_batches = 0
        for smiles, labels in self.dataloader:
            labels = labels.to(self.device)
            feats = feature_extractor(smiles).to(self.device)
            outputs = self.model(feats)
            self.model.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2)
            num_batches += 1
        for n in fisher:
            fisher[n] /= num_batches
        self.fisher = fisher

    def store_optimal_params(self):
        self.optimal_params = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
        }

    def compute_ewc_loss(self, lambda_ewc=0.1):
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]).pow(2)).sum()
        return lambda_ewc * loss

# ─── TRAIN / EVAL HELPERS ─────────────────────────────────────────────────────

def train_incremental(
    feature_extractor: nn.Module,
    classifier: nn.Module,
    train_loader: DataLoader,
    ewc: EWC,
    lambda_ewc=0.1,
    lr=1e-3
):
    feature_extractor.eval()
    classifier.train()
    opt = torch.optim.Adam(classifier.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for smiles, labels in train_loader:
        labels = labels.to(device)
        feats = feature_extractor(smiles).to(device)
        opt.zero_grad()
        out = classifier(feats)
        loss = loss_fn(out, labels)
        if ewc.fisher:
            loss += ewc.compute_ewc_loss(lambda_ewc)
        loss.backward()
        opt.step()

    # update EWC
    ewc.compute_fisher_information()
    ewc.store_optimal_params()

def evaluate_incremental(
    feature_extractor: nn.Module,
    classifier: nn.Module,
    loader: DataLoader
) -> float:
    feature_extractor.eval()
    classifier.eval()
    correct, total = 0, 0
    for smiles, labels in loader:
        labels = labels.to(device)
        feats = feature_extractor(smiles).to(device)
        preds = classifier(feats).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0

def compute_anytime_accuracy(task_acc: dict[int,list[float]]):
    sorted_tasks = sorted(task_acc)
    avg = [np.mean([task_acc[t][-1] for t in sorted_tasks[:i+1]]) 
           for i in range(len(sorted_tasks))]
    for t, a in zip(sorted_tasks, avg):
        print(f"After {t} tasks → {a:.4f}")

def compute_forgetting(task_acc: dict[int,list[float]]):
    f = [max(v)-v[-1] for v in task_acc.values()]
    fm = float(np.mean(f)) if f else 0.0
    print(f"Forgetting Measure → {fm:.4f}")

# ─── PREDICTION PIPELINE ────────────────────────────────────────────────────

feature_extractor = MoLFormerFeatureExtractor().to(device)

def initialize_pipeline_multi_dataset():
    global feature_extractor, classifier
    classifier = None
    # Load and combine all datasets
    all_data = []
    max_labels = 0
    
    for name, path in DATASETS.items():
        try:
            df = pd.read_csv(path)
            print(f"Loaded {name}: {len(df)} samples")
            all_data.append(df)
            max_labels = max(max_labels, df["Label"].nunique())
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} samples, {max_labels} classes")
    
    # Create train/val split
    train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    
    train_dataset = SMILESDataset(train_df)
    val_dataset = SMILESDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Update classifier for combined label space
    classifier = MLPClassifier(768, 256, max_labels).to(device)
    
    # Train the classifier properly
    classifier.train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    for epoch in range(50):  # More epochs
        # Training
        train_loss = 0
        for smiles, labels in train_loader:
            labels = labels.to(device)
            features = feature_extractor(smiles)
            
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            val_acc = evaluate_incremental(feature_extractor, classifier, val_loader)
            print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save best model
                torch.save({
                    'classifier_state_dict': classifier.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc
                }, 'best_model.pth')
    
    classifier.eval()
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    
    return feature_extractor, classifier

def validate_smiles(smiles: str) -> bool:
    """Validate if a SMILES string is non-empty and potentially valid"""
    if not smiles or not smiles.strip():
        return False
    # Add more validation if needed (e.g., using RDKit)
    return True

def predict_properties(smiles_list: list[str]) -> list[dict]:
    global feature_extractor, classifier
    print(f"feature_extractor type: {type(feature_extractor)}")
    print(f"feature_extractor value: {feature_extractor}")
    
    if feature_extractor is None:
        raise ValueError("Feature extractor is None. Pipeline initialization failed.")
    
    results = []
    for smiles in smiles_list:
        if not validate_smiles(smiles):
            results.append({
                "property": "predicted_label",
                "value": None,
                "confidence": 0.0,
                "error": "Invalid SMILES"
            })
            continue
            
        try:
            feats = feature_extractor([smiles]).to(device)
            out = classifier(feats)
            probs = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            
            # Better confidence metrics
            max_prob = float(probs[0].max())
            entropy = -torch.sum(probs[0] * torch.log(probs[0] + 1e-8))
            
            # Flag low-confidence predictions
            is_confident = max_prob > 0.7  # Threshold for "confident"
            
            results.append({
                "property": "predicted_label",
                "value": int(pred[0]) if is_confident else None,
                "confidence": max_prob,
                "entropy": float(entropy),
                "is_confident": is_confident,
                "raw_probabilities": probs[0].tolist()
            })
        except Exception as e:
            results.append({
                "property": "predicted_label",
                "value": None,
                "confidence": 0.0,
                "error": str(e)
            })
    
    return results

# ─── MAIN: incremental loops over each dataset ───────────────────────────────

if __name__ == "__main__":
    for name, path in DATASETS.items():
        print(f"\n=== Dataset: {name} ===")
        df = pd.read_csv(path)
        cls = MLPClassifier(768, 256, df["Label"].nunique()).to(device)
        ewc = EWC(cls, DataLoader(SMILESDataset(df), batch_size=16, shuffle=True))
        task_acc = {}

        for t in incremental_tasks:
            sel = df[df["Label"] < t]
            loader = DataLoader(SMILESDataset(sel), batch_size=16, shuffle=True)
            ewc.dataloader = loader

            train_incremental(feature_extractor, cls, loader, ewc)
            acc = evaluate_incremental(feature_extractor, cls, loader)
            task_acc.setdefault(t, []).append(acc)
            print(f" Task {t} acc = {acc:.4f}")

        compute_anytime_accuracy(task_acc)
        compute_forgetting(task_acc)
        print(f"Finished {name}\n")