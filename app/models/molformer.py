#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings as w
w.filterwarnings('ignore')

import os
base_path = os.getenv("DATA_PATH", "data")
bbbp_d = os.path.join(base_path, "BBBP.csv")
np_d = os.path.join(base_path, "NP.csv")
toxcast_d = os.path.join(base_path, "Toxcast.csv")
sider_d = os.path.join(base_path, "Sider.csv")
bitter_d = os.path.join(base_path, "explbitter.csv")
sweet_d = os.path.join(base_path, "explsweet.csv")
tox_d = os.path.join(base_path, "Tox21.csv")
clintox_d = os.path.join(base_path, "clintox.csv")

incremental_tasks = [5, 20, 25, 30]


# # EWC - NP

# In[2]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MoLFormerFeatureExtractor(nn.Module):
    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.molformer = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.molformer.eval()  # Freeze model parameters
        for param in self.molformer.parameters():
            param.requires_grad = False

    def forward(self, smiles_list):
        # Tokenize SMILES strings
        tokens = self.tokenizer(smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            # Extract features from MoLFormer
            output = self.molformer(**tokens).last_hidden_state[:, 0, :]  # Extract CLS token representation
        return output
# ✅ Trainable MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ✅ SMILES Dataset Loader
class SMILESDataset(Dataset):
    def __init__(self, dataframe):
        self.smiles = dataframe["SMILES"].values
        self.labels = dataframe["Label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]

# ✅ EWC Class
class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.fisher = {}
        self.optimal_params = {}

    def compute_fisher_information(self):
        fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}

        self.model.eval()
        for smiles, labels in self.dataloader:
            smiles, labels = list(smiles), labels.to(self.device)
            features = feature_extractor(smiles)  # Extract features
            outputs = self.model(features)  # Forward pass

            loss = nn.CrossEntropyLoss()(outputs, labels)  # Compute loss
            self.model.zero_grad()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2)

        for name in fisher:
            fisher[name] /= len(self.dataloader.dataset)

        self.fisher = fisher

    def store_optimal_params(self):
        self.optimal_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def compute_ewc_loss(self, model, lambda_ewc=0.1):
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.optimal_params[name]).pow(2)).sum()
        return lambda_ewc * loss

# ✅ Incremental Training with EWC
def train_incremental(model, classifier, train_loader, ewc, device, lambda_ewc=0.1, lr=0.001):
    model.eval()
    classifier.train()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for smiles, labels in train_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)

        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        if ewc:
            loss += ewc.compute_ewc_loss(classifier, lambda_ewc)

        loss.backward()
        optimizer.step()

    print("Updated classifier with new task data!")

    if ewc:
        ewc.compute_fisher_information()
        ewc.store_optimal_params()

# ✅ Evaluation Function
def evaluate_incremental(model, classifier, test_loader, device, task_num):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for smiles, labels in test_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)
        outputs = classifier(features)
        predictions = torch.argmax(outputs, dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"Incremental Test Accuracy on Task {task_num}: {accuracy:.4f}")

    return accuracy

# ✅ Compute Anytime Average Accuracy
task_accuracies = {}

def compute_anytime_accuracy():
    avg_accuracies = []
    sorted_tasks = sorted(task_accuracies.keys())  # Ensure task order (e.g., [5, 10, 20])

    for i in range(len(sorted_tasks)):
        current_tasks = sorted_tasks[: i + 1]  # Get all tasks up to the current one
        avg_acc = np.mean([task_accuracies[t][-1] for t in current_tasks if task_accuracies[t]])
        avg_accuracies.append(avg_acc)

    print("\n📊 Anytime Average Accuracies:")
    for t, acc in zip(sorted_tasks, avg_accuracies):
        print(f"🟢 After {t} tasks: {acc:.4f}")

    return avg_accuracies

# ✅ Compute Forgetting Measure
def compute_forgetting():
    forget_scores = []
    for label, acc_list in task_accuracies.items():
        max_acc = max(acc_list)
        last_acc = acc_list[-1]
        forget_scores.append(max_acc - last_acc)
        print("FM: ",max_acc - last_acc)

    forgetting_measure = np.mean(forget_scores)
    print(f"\n🔻 Forgetting Measure (FM): {forgetting_measure:.4f}")
    return forgetting_measure

# ✅ Load BBBP Dataset
df_bbbp = pd.read_csv(np_d)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize Feature Extractor and Trainable Classifier
feature_extractor = MoLFormerFeatureExtractor().to(device)
classifier = MLPClassifier(input_dim=768, output_dim=df_bbbp["Label"].nunique()).to(device)

# ✅ EWC Setup
ewc = EWC(classifier, None, device)

# ✅ Incremental Learning

if __name__ == "__main__":
    for task in incremental_tasks:
        print(f"\n🚀 Incremental Learning on {task} Classes")

        selected_data = df_bbbp[df_bbbp["Label"] < task]
        train_data = SMILESDataset(selected_data)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

        ewc.dataloader = train_loader  # Update EWC dataloader
        train_incremental(feature_extractor, classifier, train_loader, ewc, device)

        acc = evaluate_incremental(feature_extractor, classifier, train_loader, device, task)

        if task not in task_accuracies:
            task_accuracies[task] = []
        task_accuracies[task].append(acc)

    # ✅ Compute Anytime Average Accuracy
    compute_anytime_accuracy()

    # ✅ Compute Forgetting Measure
    compute_forgetting()

    print("\n✅ Incremental Learning with Anytime Accuracy Computation Complete!")


# # EWC - Toxcast

# In[4]:


# import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MoLFormerFeatureExtractor(nn.Module):
    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.molformer = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.molformer.eval()  # Freeze model parameters
        for param in self.molformer.parameters():
            param.requires_grad = False

    def forward(self, smiles_list):
        # Tokenize SMILES strings
        tokens = self.tokenizer(smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            # Extract features from MoLFormer
            output = self.molformer(**tokens).last_hidden_state[:, 0, :]  # Extract CLS token representation
        return output
        
# ✅ Trainable MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ✅ SMILES Dataset Loader
class SMILESDataset(Dataset):
    def __init__(self, dataframe):
        self.smiles = dataframe["SMILES"].values
        self.labels = dataframe["Label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]

# ✅ EWC Class
class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.fisher = {}
        self.optimal_params = {}

    def compute_fisher_information(self):
        fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}

        self.model.eval()
        for smiles, labels in self.dataloader:
            smiles, labels = list(smiles), labels.to(self.device)
            features = feature_extractor(smiles)  # Extract features
            outputs = self.model(features)  # Forward pass

            loss = nn.CrossEntropyLoss()(outputs, labels)  # Compute loss
            self.model.zero_grad()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2)

        for name in fisher:
            fisher[name] /= len(self.dataloader.dataset)

        self.fisher = fisher

    def store_optimal_params(self):
        self.optimal_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def compute_ewc_loss(self, model, lambda_ewc=0.1):
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.optimal_params[name]).pow(2)).sum()
        return lambda_ewc * loss

# ✅ Incremental Training with EWC
def train_incremental(model, classifier, train_loader, ewc, device, lambda_ewc=0.1, lr=0.001):
    model.eval()
    classifier.train()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for smiles, labels in train_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)

        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        if ewc:
            loss += ewc.compute_ewc_loss(classifier, lambda_ewc)

        loss.backward()
        optimizer.step()

    print("Updated classifier with new task data!")

    if ewc:
        ewc.compute_fisher_information()
        ewc.store_optimal_params()

# ✅ Evaluation Function
def evaluate_incremental(model, classifier, test_loader, device, task_num):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for smiles, labels in test_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)
        outputs = classifier(features)
        predictions = torch.argmax(outputs, dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"Incremental Test Accuracy on Task {task_num}: {accuracy:.4f}")

    return accuracy

# ✅ Compute Anytime Average Accuracy
task_accuracies = {}

def compute_anytime_accuracy():
    avg_accuracies = []
    sorted_tasks = sorted(task_accuracies.keys())  # Ensure task order (e.g., [5, 10, 20])

    for i in range(len(sorted_tasks)):
        current_tasks = sorted_tasks[: i + 1]  # Get all tasks up to the current one
        avg_acc = np.mean([task_accuracies[t][-1] for t in current_tasks if task_accuracies[t]])
        avg_accuracies.append(avg_acc)

    print("\n📊 Anytime Average Accuracies:")
    for t, acc in zip(sorted_tasks, avg_accuracies):
        print(f"🟢 After {t} tasks: {acc:.4f}")

    return avg_accuracies

# ✅ Compute Forgetting Measure
def compute_forgetting():
    forget_scores = []
    for label, acc_list in task_accuracies.items():
        max_acc = max(acc_list)
        last_acc = acc_list[-1]
        forget_scores.append(max_acc - last_acc)

    forgetting_measure = np.mean(forget_scores)
    print(f"\n🔻 Forgetting Measure (FM): {forgetting_measure:.4f}")
    return forgetting_measure

# ✅ Load BBBP Dataset
df_bbbp = pd.read_csv(toxcast_d)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize Feature Extractor and Trainable Classifier
feature_extractor = MoLFormerFeatureExtractor().to(device)
classifier = MLPClassifier(input_dim=768, output_dim=df_bbbp["Label"].nunique()).to(device)

# ✅ EWC Setup
ewc = EWC(classifier, None, device)

# ✅ Incremental Learning

for task in incremental_tasks:
    print(f"\n🚀 Incremental Learning on {task} Classes")

    selected_data = df_bbbp[df_bbbp["Label"] < task]
    train_data = SMILESDataset(selected_data)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    ewc.dataloader = train_loader  # Update EWC dataloader
    train_incremental(feature_extractor, classifier, train_loader, ewc, device)

    acc = evaluate_incremental(feature_extractor, classifier, train_loader, device, task)

    if task not in task_accuracies:
        task_accuracies[task] = []
    task_accuracies[task].append(acc)

# ✅ Compute Anytime Average Accuracy
compute_anytime_accuracy()

# ✅ Compute Forgetting Measure
compute_forgetting()

print("\n✅ Incremental Learning with Anytime Accuracy Computation Complete!")


# # EWC - Sider

# In[7]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MoLFormerFeatureExtractor(nn.Module):
    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.molformer = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.molformer.eval()  # Freeze model parameters
        for param in self.molformer.parameters():
            param.requires_grad = False

    def forward(self, smiles_list):
        # Tokenize SMILES strings
        tokens = self.tokenizer(smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            # Extract features from MoLFormer
            output = self.molformer(**tokens).last_hidden_state[:, 0, :]  # Extract CLS token representation
        return output

# ✅ Trainable MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ✅ SMILES Dataset Loader
class SMILESDataset(Dataset):
    def __init__(self, dataframe):
        self.smiles = dataframe["SMILES"].values
        self.labels = dataframe["Label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]

# ✅ EWC Class
class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.fisher = {}
        self.optimal_params = {}

    def compute_fisher_information(self):
        fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}

        self.model.eval()
        for smiles, labels in self.dataloader:
            smiles, labels = list(smiles), labels.to(self.device)
            features = feature_extractor(smiles)  # Extract features
            outputs = self.model(features)  # Forward pass

            loss = nn.CrossEntropyLoss()(outputs, labels)  # Compute loss
            self.model.zero_grad()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2)

        for name in fisher:
            fisher[name] /= len(self.dataloader.dataset)

        self.fisher = fisher

    def store_optimal_params(self):
        self.optimal_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def compute_ewc_loss(self, model, lambda_ewc=0.1):
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.optimal_params[name]).pow(2)).sum()
        return lambda_ewc * loss

# ✅ Incremental Training with EWC
def train_incremental(model, classifier, train_loader, ewc, device, lambda_ewc=0.1, lr=0.001):
    model.eval()
    classifier.train()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for smiles, labels in train_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)

        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        if ewc:
            loss += ewc.compute_ewc_loss(classifier, lambda_ewc)

        loss.backward()
        optimizer.step()

    print("Updated classifier with new task data!")

    if ewc:
        ewc.compute_fisher_information()
        ewc.store_optimal_params()

# ✅ Evaluation Function
def evaluate_incremental(model, classifier, test_loader, device, task_num):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for smiles, labels in test_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)
        outputs = classifier(features)
        predictions = torch.argmax(outputs, dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"Incremental Test Accuracy on Task {task_num}: {accuracy:.4f}")

    return accuracy

# ✅ Compute Anytime Average Accuracy
task_accuracies = {}

def compute_anytime_accuracy():
    avg_accuracies = []
    sorted_tasks = sorted(task_accuracies.keys())  # Ensure task order (e.g., [5, 10, 20])

    for i in range(len(sorted_tasks)):
        current_tasks = sorted_tasks[: i + 1]  # Get all tasks up to the current one
        avg_acc = np.mean([task_accuracies[t][-1] for t in current_tasks if task_accuracies[t]])
        avg_accuracies.append(avg_acc)

    print("\n📊 Anytime Average Accuracies:")
    for t, acc in zip(sorted_tasks, avg_accuracies):
        print(f"🟢 After {t} tasks: {acc:.4f}")

    return avg_accuracies

# ✅ Compute Forgetting Measure
def compute_forgetting():
    forget_scores = []
    for label, acc_list in task_accuracies.items():
        max_acc = max(acc_list)
        last_acc = acc_list[-1]
        forget_scores.append(max_acc - last_acc)

    forgetting_measure = np.mean(forget_scores)
    print(f"\n🔻 Forgetting Measure (FM): {forgetting_measure:.4f}")
    return forgetting_measure

# ✅ Load BBBP Dataset
df_bbbp = pd.read_csv(sider_d)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize Feature Extractor and Trainable Classifier
feature_extractor = MoLFormerFeatureExtractor().to(device)
classifier = MLPClassifier(input_dim=768, output_dim=df_bbbp["Label"].nunique()).to(device)

# ✅ EWC Setup
ewc = EWC(classifier, None, device)

# ✅ Incremental Learning

for task in incremental_tasks:
    print(f"\n🚀 Incremental Learning on {task} Classes")

    selected_data = df_bbbp[df_bbbp["Label"] < task]
    train_data = SMILESDataset(selected_data)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    ewc.dataloader = train_loader  # Update EWC dataloader
    train_incremental(feature_extractor, classifier, train_loader, ewc, device)

    acc = evaluate_incremental(feature_extractor, classifier, train_loader, device, task)

    if task not in task_accuracies:
        task_accuracies[task] = []
    task_accuracies[task].append(acc)

# ✅ Compute Anytime Average Accuracy
compute_anytime_accuracy()

# ✅ Compute Forgetting Measure
compute_forgetting()

print("\n✅ Incremental Learning with Anytime Accuracy Computation Complete!")


# # EWC - BBBP

# In[8]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MoLFormerFeatureExtractor(nn.Module):
    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.molformer = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.molformer.eval()  # Freeze model parameters
        for param in self.molformer.parameters():
            param.requires_grad = False

    def forward(self, smiles_list):
        # Tokenize SMILES strings
        tokens = self.tokenizer(smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            # Extract features from MoLFormer
            output = self.molformer(**tokens).last_hidden_state[:, 0, :]  # Extract CLS token representation
        return output

# ✅ Trainable MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ✅ SMILES Dataset Loader
class SMILESDataset(Dataset):
    def __init__(self, dataframe):
        self.smiles = dataframe["SMILES"].values
        self.labels = dataframe["Label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]

# ✅ EWC Class
class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.fisher = {}
        self.optimal_params = {}

    def compute_fisher_information(self):
        fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}

        self.model.eval()
        for smiles, labels in self.dataloader:
            smiles, labels = list(smiles), labels.to(self.device)
            features = feature_extractor(smiles)  # Extract features
            outputs = self.model(features)  # Forward pass

            loss = nn.CrossEntropyLoss()(outputs, labels)  # Compute loss
            self.model.zero_grad()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2)

        for name in fisher:
            fisher[name] /= len(self.dataloader.dataset)

        self.fisher = fisher

    def store_optimal_params(self):
        self.optimal_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def compute_ewc_loss(self, model, lambda_ewc=0.1):
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.optimal_params[name]).pow(2)).sum()
        return lambda_ewc * loss

# ✅ Incremental Training with EWC
def train_incremental(model, classifier, train_loader, ewc, device, lambda_ewc=0.1, lr=0.001):
    model.eval()
    classifier.train()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for smiles, labels in train_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)

        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        if ewc:
            loss += ewc.compute_ewc_loss(classifier, lambda_ewc)

        loss.backward()
        optimizer.step()

    print("Updated classifier with new task data!")

    if ewc:
        ewc.compute_fisher_information()
        ewc.store_optimal_params()

# ✅ Evaluation Function
def evaluate_incremental(model, classifier, test_loader, device, task_num):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for smiles, labels in test_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)
        outputs = classifier(features)
        predictions = torch.argmax(outputs, dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"Incremental Test Accuracy on Task {task_num}: {accuracy:.4f}")

    return accuracy

# ✅ Compute Anytime Average Accuracy
task_accuracies = {}

def compute_anytime_accuracy():
    avg_accuracies = []
    sorted_tasks = sorted(task_accuracies.keys())  # Ensure task order (e.g., [5, 10, 20])

    for i in range(len(sorted_tasks)):
        current_tasks = sorted_tasks[: i + 1]  # Get all tasks up to the current one
        avg_acc = np.mean([task_accuracies[t][-1] for t in current_tasks if task_accuracies[t]])
        avg_accuracies.append(avg_acc)

    print("\n📊 Anytime Average Accuracies:")
    for t, acc in zip(sorted_tasks, avg_accuracies):
        print(f"🟢 After {t} tasks: {acc:.4f}")

    return avg_accuracies

# ✅ Compute Forgetting Measure
def compute_forgetting():
    forget_scores = []
    for label, acc_list in task_accuracies.items():
        max_acc = max(acc_list)
        last_acc = acc_list[-1]
        forget_scores.append(max_acc - last_acc)

    forgetting_measure = np.mean(forget_scores)
    print(f"\n🔻 Forgetting Measure (FM): {forgetting_measure:.4f}")
    return forgetting_measure

# ✅ Load BBBP Dataset
df_bbbp = pd.read_csv(bbbp_d)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize Feature Extractor and Trainable Classifier
feature_extractor = MoLFormerFeatureExtractor().to(device)
classifier = MLPClassifier(input_dim=768, output_dim=df_bbbp["Label"].nunique()).to(device)

# ✅ EWC Setup
ewc = EWC(classifier, None, device)

# ✅ Incremental Learning

for task in incremental_tasks:
    print(f"\n🚀 Incremental Learning on {task} Classes")

    selected_data = df_bbbp[df_bbbp["Label"] < task]
    train_data = SMILESDataset(selected_data)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    ewc.dataloader = train_loader  # Update EWC dataloader
    train_incremental(feature_extractor, classifier, train_loader, ewc, device)

    acc = evaluate_incremental(feature_extractor, classifier, train_loader, device, task)

    if task not in task_accuracies:
        task_accuracies[task] = []
    task_accuracies[task].append(acc)

# ✅ Compute Anytime Average Accuracy
compute_anytime_accuracy()

# ✅ Compute Forgetting Measure
compute_forgetting()

print("\n✅ Incremental Learning with Anytime Accuracy Computation Complete!")


# # EWC - Sweet

# In[9]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MoLFormerFeatureExtractor(nn.Module):
    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.molformer = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.molformer.eval()  # Freeze model parameters
        for param in self.molformer.parameters():
            param.requires_grad = False

    def forward(self, smiles_list):
        # Tokenize SMILES strings
        tokens = self.tokenizer(smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            # Extract features from MoLFormer
            output = self.molformer(**tokens).last_hidden_state[:, 0, :]  # Extract CLS token representation
        return output

# ✅ Trainable MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ✅ SMILES Dataset Loader
class SMILESDataset(Dataset):
    def __init__(self, dataframe):
        self.smiles = dataframe["SMILES"].values
        self.labels = dataframe["Label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]

# ✅ EWC Class
class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.fisher = {}
        self.optimal_params = {}

    def compute_fisher_information(self):
        fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}

        self.model.eval()
        for smiles, labels in self.dataloader:
            smiles, labels = list(smiles), labels.to(self.device)
            features = feature_extractor(smiles)  # Extract features
            outputs = self.model(features)  # Forward pass

            loss = nn.CrossEntropyLoss()(outputs, labels)  # Compute loss
            self.model.zero_grad()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2)

        for name in fisher:
            fisher[name] /= len(self.dataloader.dataset)

        self.fisher = fisher

    def store_optimal_params(self):
        self.optimal_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def compute_ewc_loss(self, model, lambda_ewc=0.1):
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.optimal_params[name]).pow(2)).sum()
        return lambda_ewc * loss

# ✅ Incremental Training with EWC
def train_incremental(model, classifier, train_loader, ewc, device, lambda_ewc=0.1, lr=0.001):
    model.eval()
    classifier.train()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for smiles, labels in train_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)

        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        if ewc:
            loss += ewc.compute_ewc_loss(classifier, lambda_ewc)

        loss.backward()
        optimizer.step()

    print("Updated classifier with new task data!")

    if ewc:
        ewc.compute_fisher_information()
        ewc.store_optimal_params()

# ✅ Evaluation Function
def evaluate_incremental(model, classifier, test_loader, device, task_num):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for smiles, labels in test_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)
        outputs = classifier(features)
        predictions = torch.argmax(outputs, dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"Incremental Test Accuracy on Task {task_num}: {accuracy:.4f}")

    return accuracy

# ✅ Compute Anytime Average Accuracy
task_accuracies = {}

def compute_anytime_accuracy():
    avg_accuracies = []
    sorted_tasks = sorted(task_accuracies.keys())  # Ensure task order (e.g., [5, 10, 20])

    for i in range(len(sorted_tasks)):
        current_tasks = sorted_tasks[: i + 1]  # Get all tasks up to the current one
        avg_acc = np.mean([task_accuracies[t][-1] for t in current_tasks if task_accuracies[t]])
        avg_accuracies.append(avg_acc)

    print("\n📊 Anytime Average Accuracies:")
    for t, acc in zip(sorted_tasks, avg_accuracies):
        print(f"🟢 After {t} tasks: {acc:.4f}")

    return avg_accuracies

# ✅ Compute Forgetting Measure
def compute_forgetting():
    forget_scores = []
    for label, acc_list in task_accuracies.items():
        max_acc = max(acc_list)
        last_acc = acc_list[-1]
        forget_scores.append(max_acc - last_acc)

    forgetting_measure = np.mean(forget_scores)
    print(f"\n🔻 Forgetting Measure (FM): {forgetting_measure:.4f}")
    return forgetting_measure

# ✅ Load BBBP Dataset
df_bbbp = pd.read_csv(sweet_d)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize Feature Extractor and Trainable Classifier
feature_extractor = MoLFormerFeatureExtractor().to(device)
classifier = MLPClassifier(input_dim=768, output_dim=df_bbbp["Label"].nunique()).to(device)

# ✅ EWC Setup
ewc = EWC(classifier, None, device)

# ✅ Incremental Learning

for task in incremental_tasks:
    print(f"\n🚀 Incremental Learning on {task} Classes")

    selected_data = df_bbbp[df_bbbp["Label"] < task]
    train_data = SMILESDataset(selected_data)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    ewc.dataloader = train_loader  # Update EWC dataloader
    train_incremental(feature_extractor, classifier, train_loader, ewc, device)

    acc = evaluate_incremental(feature_extractor, classifier, train_loader, device, task)

    if task not in task_accuracies:
        task_accuracies[task] = []
    task_accuracies[task].append(acc)

# ✅ Compute Anytime Average Accuracy
compute_anytime_accuracy()

# ✅ Compute Forgetting Measure
compute_forgetting()

print("\n✅ Incremental Learning with Anytime Accuracy Computation Complete!")


# # EWC - Bitter

# In[10]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MoLFormerFeatureExtractor(nn.Module):
    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.molformer = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.molformer.eval()  # Freeze model parameters
        for param in self.molformer.parameters():
            param.requires_grad = False

    def forward(self, smiles_list):
        # Tokenize SMILES strings
        tokens = self.tokenizer(smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            # Extract features from MoLFormer
            output = self.molformer(**tokens).last_hidden_state[:, 0, :]  # Extract CLS token representation
        return output

# ✅ Trainable MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ✅ SMILES Dataset Loader
class SMILESDataset(Dataset):
    def __init__(self, dataframe):
        self.smiles = dataframe["SMILES"].values
        self.labels = dataframe["Label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]

# ✅ EWC Class
class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.fisher = {}
        self.optimal_params = {}

    def compute_fisher_information(self):
        fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}

        self.model.eval()
        for smiles, labels in self.dataloader:
            smiles, labels = list(smiles), labels.to(self.device)
            features = feature_extractor(smiles)  # Extract features
            outputs = self.model(features)  # Forward pass

            loss = nn.CrossEntropyLoss()(outputs, labels)  # Compute loss
            self.model.zero_grad()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2)

        for name in fisher:
            fisher[name] /= len(self.dataloader.dataset)

        self.fisher = fisher

    def store_optimal_params(self):
        self.optimal_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def compute_ewc_loss(self, model, lambda_ewc=0.1):
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.optimal_params[name]).pow(2)).sum()
        return lambda_ewc * loss

# ✅ Incremental Training with EWC
def train_incremental(model, classifier, train_loader, ewc, device, lambda_ewc=0.1, lr=0.001):
    model.eval()
    classifier.train()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for smiles, labels in train_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)

        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        if ewc:
            loss += ewc.compute_ewc_loss(classifier, lambda_ewc)

        loss.backward()
        optimizer.step()

    print("Updated classifier with new task data!")

    if ewc:
        ewc.compute_fisher_information()
        ewc.store_optimal_params()

# ✅ Evaluation Function
def evaluate_incremental(model, classifier, test_loader, device, task_num):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for smiles, labels in test_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)
        outputs = classifier(features)
        predictions = torch.argmax(outputs, dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"Incremental Test Accuracy on Task {task_num}: {accuracy:.4f}")

    return accuracy

# ✅ Compute Anytime Average Accuracy
task_accuracies = {}

def compute_anytime_accuracy():
    avg_accuracies = []
    sorted_tasks = sorted(task_accuracies.keys())  # Ensure task order (e.g., [5, 10, 20])

    for i in range(len(sorted_tasks)):
        current_tasks = sorted_tasks[: i + 1]  # Get all tasks up to the current one
        avg_acc = np.mean([task_accuracies[t][-1] for t in current_tasks if task_accuracies[t]])
        avg_accuracies.append(avg_acc)

    print("\n📊 Anytime Average Accuracies:")
    for t, acc in zip(sorted_tasks, avg_accuracies):
        print(f"🟢 After {t} tasks: {acc:.4f}")

    return avg_accuracies

# ✅ Compute Forgetting Measure
def compute_forgetting():
    forget_scores = []
    for label, acc_list in task_accuracies.items():
        max_acc = max(acc_list)
        last_acc = acc_list[-1]
        forget_scores.append(max_acc - last_acc)

    forgetting_measure = np.mean(forget_scores)
    print(f"\n🔻 Forgetting Measure (FM): {forgetting_measure:.4f}")
    return forgetting_measure

# ✅ Load BBBP Dataset
df_bbbp = pd.read_csv(bitter_d)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize Feature Extractor and Trainable Classifier
feature_extractor = MoLFormerFeatureExtractor().to(device)
classifier = MLPClassifier(input_dim=768, output_dim=df_bbbp["Label"].nunique()).to(device)

# ✅ EWC Setup
ewc = EWC(classifier, None, device)

# ✅ Incremental Learning

for task in incremental_tasks:
    print(f"\n🚀 Incremental Learning on {task} Classes")

    selected_data = df_bbbp[df_bbbp["Label"] < task]
    train_data = SMILESDataset(selected_data)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    ewc.dataloader = train_loader  # Update EWC dataloader
    train_incremental(feature_extractor, classifier, train_loader, ewc, device)

    acc = evaluate_incremental(feature_extractor, classifier, train_loader, device, task)

    if task not in task_accuracies:
        task_accuracies[task] = []
    task_accuracies[task].append(acc)

# ✅ Compute Anytime Average Accuracy
compute_anytime_accuracy()

# ✅ Compute Forgetting Measure
compute_forgetting()

print("\n✅ Incremental Learning with Anytime Accuracy Computation Complete!")


# # EWC - Tox

# In[3]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MoLFormerFeatureExtractor(nn.Module):
    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.molformer = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.molformer.eval()  # Freeze model parameters
        for param in self.molformer.parameters():
            param.requires_grad = False

    def forward(self, smiles_list):
        # Tokenize SMILES strings
        tokens = self.tokenizer(smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            # Extract features from MoLFormer
            output = self.molformer(**tokens).last_hidden_state[:, 0, :]  # Extract CLS token representation
        return output

# ✅ Trainable MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ✅ SMILES Dataset Loader
class SMILESDataset(Dataset):
    def __init__(self, dataframe):
        self.smiles = dataframe["SMILES"].values
        self.labels = dataframe["Label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]

# ✅ EWC Class
class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.fisher = {}
        self.optimal_params = {}

    def compute_fisher_information(self):
        fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}

        self.model.eval()
        for smiles, labels in self.dataloader:
            smiles, labels = list(smiles), labels.to(self.device)
            features = feature_extractor(smiles)  # Extract features
            outputs = self.model(features)  # Forward pass

            loss = nn.CrossEntropyLoss()(outputs, labels)  # Compute loss
            self.model.zero_grad()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2)

        for name in fisher:
            fisher[name] /= len(self.dataloader.dataset)

        self.fisher = fisher

    def store_optimal_params(self):
        self.optimal_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def compute_ewc_loss(self, model, lambda_ewc=0.1):
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.optimal_params[name]).pow(2)).sum()
        return lambda_ewc * loss

# ✅ Incremental Training with EWC
def train_incremental(model, classifier, train_loader, ewc, device, lambda_ewc=0.1, lr=0.001):
    model.eval()
    classifier.train()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for smiles, labels in train_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)

        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        if ewc:
            loss += ewc.compute_ewc_loss(classifier, lambda_ewc)

        loss.backward()
        optimizer.step()

    print("Updated classifier with new task data!")

    if ewc:
        ewc.compute_fisher_information()
        ewc.store_optimal_params()

# ✅ Evaluation Function
def evaluate_incremental(model, classifier, test_loader, device, task_num):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for smiles, labels in test_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)
        outputs = classifier(features)
        predictions = torch.argmax(outputs, dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"Incremental Test Accuracy on Task {task_num}: {accuracy:.4f}")

    return accuracy

# ✅ Compute Anytime Average Accuracy
task_accuracies = {}

def compute_anytime_accuracy():
    avg_accuracies = []
    sorted_tasks = sorted(task_accuracies.keys())  # Ensure task order (e.g., [5, 10, 20])

    for i in range(len(sorted_tasks)):
        current_tasks = sorted_tasks[: i + 1]  # Get all tasks up to the current one
        avg_acc = np.mean([task_accuracies[t][-1] for t in current_tasks if task_accuracies[t]])
        avg_accuracies.append(avg_acc)

    print("\n📊 Anytime Average Accuracies:")
    for t, acc in zip(sorted_tasks, avg_accuracies):
        print(f"🟢 After {t} tasks: {acc:.4f}")

    return avg_accuracies

# ✅ Compute Forgetting Measure
def compute_forgetting():
    forget_scores = []
    for label, acc_list in task_accuracies.items():
        max_acc = max(acc_list)
        last_acc = acc_list[-1]
        forget_scores.append(max_acc - last_acc)

    forgetting_measure = np.mean(forget_scores)
    print(f"\n🔻 Forgetting Measure (FM): {forgetting_measure:.4f}")
    return forgetting_measure

# ✅ Load BBBP Dataset
df_bbbp = pd.read_csv(tox_d)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize Feature Extractor and Trainable Classifier
feature_extractor = MoLFormerFeatureExtractor().to(device)
classifier = MLPClassifier(input_dim=768, output_dim=df_bbbp["Label"].nunique()).to(device)

# ✅ EWC Setup
ewc = EWC(classifier, None, device)

# ✅ Incremental Learning

for task in incremental_tasks:
    print(f"\n🚀 Incremental Learning on {task} Classes")

    selected_data = df_bbbp[df_bbbp["Label"] < task]
    train_data = SMILESDataset(selected_data)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    ewc.dataloader = train_loader  # Update EWC dataloader
    train_incremental(feature_extractor, classifier, train_loader, ewc, device)

    acc = evaluate_incremental(feature_extractor, classifier, train_loader, device, task)

    if task not in task_accuracies:
        task_accuracies[task] = []
    task_accuracies[task].append(acc)

# ✅ Compute Anytime Average Accuracy
compute_anytime_accuracy()

# ✅ Compute Forgetting Measure
compute_forgetting()

print("\n✅ Incremental Learning with Anytime Accuracy Computation Complete!")


# # EWC - Clintox

# In[4]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MoLFormerFeatureExtractor(nn.Module):
    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.molformer = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.molformer.eval()  # Freeze model parameters
        for param in self.molformer.parameters():
            param.requires_grad = False

    def forward(self, smiles_list):
        # Tokenize SMILES strings
        tokens = self.tokenizer(smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            # Extract features from MoLFormer
            output = self.molformer(**tokens).last_hidden_state[:, 0, :]  # Extract CLS token representation
        return output

# ✅ Trainable MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ✅ SMILES Dataset Loader
class SMILESDataset(Dataset):
    def __init__(self, dataframe):
        self.smiles = dataframe["SMILES"].values
        self.labels = dataframe["Label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]

# ✅ EWC Class
class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.fisher = {}
        self.optimal_params = {}

    def compute_fisher_information(self):
        fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}

        self.model.eval()
        for smiles, labels in self.dataloader:
            smiles, labels = list(smiles), labels.to(self.device)
            features = feature_extractor(smiles)  # Extract features
            outputs = self.model(features)  # Forward pass

            loss = nn.CrossEntropyLoss()(outputs, labels)  # Compute loss
            self.model.zero_grad()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2)

        for name in fisher:
            fisher[name] /= len(self.dataloader.dataset)

        self.fisher = fisher

    def store_optimal_params(self):
        self.optimal_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def compute_ewc_loss(self, model, lambda_ewc=0.1):
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.optimal_params[name]).pow(2)).sum()
        return lambda_ewc * loss

# ✅ Incremental Training with EWC
def train_incremental(model, classifier, train_loader, ewc, device, lambda_ewc=0.1, lr=0.001):
    model.eval()
    classifier.train()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for smiles, labels in train_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)

        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        if ewc:
            loss += ewc.compute_ewc_loss(classifier, lambda_ewc)

        loss.backward()
        optimizer.step()

    print("Updated classifier with new task data!")

    if ewc:
        ewc.compute_fisher_information()
        ewc.store_optimal_params()

# ✅ Evaluation Function
def evaluate_incremental(model, classifier, test_loader, device, task_num):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for smiles, labels in test_loader:
        smiles, labels = list(smiles), labels.to(device)
        features = model(smiles).to(device)
        outputs = classifier(features)
        predictions = torch.argmax(outputs, dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"Incremental Test Accuracy on Task {task_num}: {accuracy:.4f}")

    return accuracy

# ✅ Compute Anytime Average Accuracy
task_accuracies = {}

def compute_anytime_accuracy():
    avg_accuracies = []
    sorted_tasks = sorted(task_accuracies.keys())  # Ensure task order (e.g., [5, 10, 20])

    for i in range(len(sorted_tasks)):
        current_tasks = sorted_tasks[: i + 1]  # Get all tasks up to the current one
        avg_acc = np.mean([task_accuracies[t][-1] for t in current_tasks if task_accuracies[t]])
        avg_accuracies.append(avg_acc)

    print("\n📊 Anytime Average Accuracies:")
    for t, acc in zip(sorted_tasks, avg_accuracies):
        print(f"🟢 After {t} tasks: {acc:.4f}")

    return avg_accuracies

# ✅ Compute Forgetting Measure
def compute_forgetting():
    forget_scores = []
    for label, acc_list in task_accuracies.items():
        max_acc = max(acc_list)
        last_acc = acc_list[-1]
        forget_scores.append(max_acc - last_acc)

    forgetting_measure = np.mean(forget_scores)
    print(f"\n🔻 Forgetting Measure (FM): {forgetting_measure:.4f}")
    return forgetting_measure

# ✅ Load BBBP Dataset
df_bbbp = pd.read_csv(clintox_d)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize Feature Extractor and Trainable Classifier
feature_extractor = MoLFormerFeatureExtractor().to(device)
classifier = MLPClassifier(input_dim=768, output_dim=df_bbbp["Label"].nunique()).to(device)

# ✅ EWC Setup
ewc = EWC(classifier, None, device)

# ✅ Incremental Learning

for task in incremental_tasks:
    print(f"\n🚀 Incremental Learning on {task} Classes")

    selected_data = df_bbbp[df_bbbp["Label"] < task]
    train_data = SMILESDataset(selected_data)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    ewc.dataloader = train_loader  # Update EWC dataloader
    train_incremental(feature_extractor, classifier, train_loader, ewc, device)

    acc = evaluate_incremental(feature_extractor, classifier, train_loader, device, task)

    if task not in task_accuracies:
        task_accuracies[task] = []
    task_accuracies[task].append(acc)

# ✅ Compute Anytime Average Accuracy
compute_anytime_accuracy()

# ✅ Compute Forgetting Measure
compute_forgetting()

print("\n✅ Incremental Learning with Anytime Accuracy Computation Complete!")


# In[ ]:





# --- At the end of the file, add the pipeline initialization and prediction logic ---

def initialize_pipeline():
    global feature_extractor, classifier
    feature_extractor = MoLFormerFeatureExtractor().to(device)
    dummy_df = pd.read_csv(np_d)
    classifier = MLPClassifier(input_dim=768, output_dim=dummy_df["Label"].nunique()).to(device)
    classifier.eval()
    return feature_extractor, classifier

def predict_properties(smiles_list: list[str]) -> list[dict]:
    features = feature_extractor(smiles_list).to(device)
    outputs = classifier(features)
    probs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)
    return [
        {
            "property": "predicted_label",
            "value": str(pred.item()),
            "confidence": float(torch.max(prob).item())
        }
        for pred, prob in zip(preds, probs)
    ]