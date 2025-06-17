#!/usr/bin/env python
# coding: utf-8
from typing import Any, Dict
# In[1]:

dataset_forgetting: Dict[str, Any] = {}
import warnings as w
w.filterwarnings('ignore')

bbbp_d = "C:/Chaitanya/MolFECAM/MolFECAM-Backend/data/BBBP.csv"
np_d = "C:/Chaitanya/MolFECAM/MolFECAM-Backend/data/NP.csv"
toxcast_d = "C:/Chaitanya/MolFECAM/MolFECAM-Backend/data/Toxcast.csv"
sider_d = "C:/Chaitanya/MolFECAM/MolFECAM-Backend/data/Sider.csv"
bitter_d = "C:/Chaitanya/MolFECAM/MolFECAM-Backend/data/explbitter.csv"
sweet_d = "C:/Chaitanya/MolFECAM/MolFECAM-Backend/data/explsweet.csv"
tox_d = "C:/Chaitanya/MolFECAM/MolFECAM-Backend/data/Tox21.csv"
clintox_d = "C:/Chaitanya/MolFECAM/MolFECAM-Backend/data/clintox.csv"

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
fm_np = compute_forgetting()
dataset_forgetting["NP"] = fm_np
print("\n✅ NP Incremental Learning with Anytime Accuracy Computation Complete!")


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
fm_toxcast = compute_forgetting()
dataset_forgetting["Toxcast"] = fm_toxcast
print("\n✅ ToxCast Incremental Learning with Anytime Accuracy Computation Complete!")


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
fm_sider = compute_forgetting()
dataset_forgetting["Sider"] = fm_sider
print("\n✅ Sider Incremental Learning with Anytime Accuracy Computation Complete!")


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
fm_bbbp = compute_forgetting()
dataset_forgetting["BBBP"] = fm_bbbp
print("\n✅ BBBP Incremental Learning with Anytime Accuracy Computation Complete!")


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
    def __init__(self, input_dim, hidden_dim=256, num_labels=2, num_tastes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_labels)
        self.fc3 = nn.Linear(hidden_dim, num_tastes)

    def forward(self, x):
        h          = self.relu(self.fc1(x))
        logits_lbl = self.fc2(h)
        logits_taste = self.fc3(h)
        return logits_lbl, logits_taste

# ✅ SMILES Dataset Loader
class SMILESDataset(Dataset):
    def __init__(self, dataframe):
        self.smiles = dataframe["SMILES"].values
        self.labels = dataframe["Label"].values
        # factorize returns (codes, unique_labels_in_order)
        taste_codes, taste_strings = pd.factorize(dataframe["Taste"])
        self.tastes = taste_codes
        # store mapping once
        self.taste_labels = taste_strings.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        smi   = self.smiles[idx]
        lbl   = torch.tensor(self.labels[idx], dtype=torch.long)
        taste = torch.tensor(self.tastes[idx], dtype=torch.long)
        return smi, lbl, taste
    
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
        for smiles, labels, tastes in self.dataloader:
            smiles, labels, tastes = list(smiles), labels.to(self.device), tastes.to(self.device)
            features = feature_extractor(smiles)  # Extract features
            out_lbl, out_taste = self.model(features)  # Forward pass

            loss = nn.CrossEntropyLoss()(out_lbl, labels) + nn.CrossEntropyLoss()(out_taste, tastes)  # Compute loss
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

    for smiles, labels, tastes in train_loader:
        smiles, labels, tastes = list(smiles), labels.to(device), tastes.to(device)
        features = model(smiles).to(device)

        optimizer.zero_grad()
        out_lbl, out_taste = classifier(features)
        loss = criterion(out_lbl, labels) + criterion(out_taste, tastes)

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
    correct_lbl = 0
    correct_taste = 0
    total = 0
    all_preds = []
    all_labels = []

    for smiles, labels, tastes in test_loader:
        smiles, labels, tastes = list(smiles), labels.to(device), tastes.to(device)
        features = model(smiles).to(device)
        out_lbl, out_taste = classifier(features)
        predictions_lbl = torch.argmax(out_lbl, dim=1)
        predictions_taste = torch.argmax(out_taste, dim=1)

        all_preds.extend(predictions_lbl.cpu().numpy() + predictions_taste.cpu().numpy()) 
        all_labels.extend(labels.cpu().numpy())

        correct_lbl += (predictions_lbl == labels).sum().item()
        correct_taste += (predictions_taste == tastes).sum().item()
        total += labels.size(0)

    accuracy_lbl = correct_lbl / total
    accuracy_taste = correct_taste / total
    print(f"Incremental Test Accuracy on Task {task_num}: {accuracy_lbl:.4f} | {accuracy_taste:.4f}")

    return accuracy_lbl, accuracy_taste

# ✅ Compute Anytime Average Accuracy
task_accuracies = {}

def compute_anytime_accuracy():
    avg_accuracies_lbl = []
    avg_accuracies_taste = []
    sorted_tasks = sorted(task_accuracies.keys())  # Ensure task order (e.g., [5, 10, 20])

    for i in range(len(sorted_tasks)):
        current_tasks = sorted_tasks[: i + 1]  # Get all tasks up to the current one
        avg_acc_lbl = np.mean([task_accuracies[t][-1][0] for t in current_tasks if task_accuracies[t]])
        avg_acc_taste = np.mean([task_accuracies[t][-1][1] for t in current_tasks if task_accuracies[t]])
        avg_accuracies_lbl.append(avg_acc_lbl)
        avg_accuracies_taste.append(avg_acc_taste)

    print("\n📊 Anytime Average Accuracies:")
    for t, acc_lbl, acc_taste in zip(sorted_tasks, avg_accuracies_lbl, avg_accuracies_taste):
        print(f"🟢 After {t} tasks: {acc_lbl:.4f}, {acc_taste:.4f}")

    return avg_accuracies_lbl, avg_accuracies_taste

# ✅ Compute Forgetting Measure
def compute_forgetting():
    forget_scores_lbl, forget_scores_taste = [], []
    for label, acc_list in task_accuracies.items():
        max_acc_lbl = max(acc[0] for acc in acc_list)
        last_acc_lbl = acc_list[-1][0]
        forget_scores_lbl.append(max_acc_lbl - last_acc_lbl)

    forgetting_measure_lbl = np.mean(forget_scores_lbl)
    print(f"\n🔻 Forgetting Measure (FM) - Label: {forgetting_measure_lbl:.4f}")

    for tastes, acc_list in task_accuracies.items():
        max_acc_taste = max(acc[1] for acc in acc_list)
        last_acc_taste = acc_list[-1][1]
        forget_scores_taste.append(max_acc_taste - last_acc_taste)

    forgetting_measure_taste = np.mean(forget_scores_taste)
    print(f"\n🔻 Forgetting Measure (FM) - Taste: {forgetting_measure_taste:.4f}")

    return forgetting_measure_lbl, forgetting_measure_taste

# ✅ Load BBBP Dataset
df_bbbp = pd.read_csv(sweet_d)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize Feature Extractor and Trainable Classifier
feature_extractor = MoLFormerFeatureExtractor().to(device)
classifier = MLPClassifier(input_dim=768, num_labels=df_bbbp["Label"].nunique(), num_tastes=df_bbbp["Taste"].nunique()).to(device)

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
fm_sweet = compute_forgetting()
dataset_forgetting["Sweet"] = fm_sweet
print("\n✅ Sweet Incremental Learning with Anytime Accuracy Computation Complete!")


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
    def __init__(self, input_dim, hidden_dim=256, num_labels=2, num_tastes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_labels)
        self.fc3 = nn.Linear(hidden_dim, num_tastes)

    def forward(self, x):
        h          = self.relu(self.fc1(x))
        logits_lbl = self.fc2(h)
        logits_taste = self.fc3(h)
        return logits_lbl, logits_taste

# ✅ SMILES Dataset Loader
class SMILESDataset(Dataset):
    def __init__(self, dataframe):
        self.smiles = dataframe["SMILES"].values
        self.labels = dataframe["Label"].values
        # factorize returns (codes, unique_labels_in_order)
        taste_codes, taste_strings = pd.factorize(dataframe["Taste"])
        self.tastes = taste_codes
        # store mapping once
        self.taste_labels = taste_strings.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        smi   = self.smiles[idx]
        lbl   = torch.tensor(self.labels[idx], dtype=torch.long)
        taste = torch.tensor(self.tastes[idx], dtype=torch.long)
        return smi, lbl, taste
    
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
        for smiles, labels, tastes in self.dataloader:
            smiles, labels, tastes = list(smiles), labels.to(self.device), tastes.to(self.device)
            features = feature_extractor(smiles)  # Extract features
            out_lbl, out_taste = self.model(features)  # Forward pass

            loss = nn.CrossEntropyLoss()(out_lbl, labels) + nn.CrossEntropyLoss()(out_taste, tastes)  # Compute loss
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

    for smiles, labels, tastes in train_loader:
        smiles, labels, tastes = list(smiles), labels.to(device), tastes.to(device)
        features = model(smiles).to(device)

        optimizer.zero_grad()
        out_lbl, out_taste = classifier(features)
        loss = criterion(out_lbl, labels) + criterion(out_taste, tastes)

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
    correct_lbl = 0
    correct_taste = 0
    total = 0
    all_preds = []
    all_labels = []

    for smiles, labels, tastes in test_loader:
        smiles, labels, tastes = list(smiles), labels.to(device), tastes.to(device)
        features = model(smiles).to(device)
        out_lbl, out_taste = classifier(features)
        predictions_lbl = torch.argmax(out_lbl, dim=1)
        predictions_taste = torch.argmax(out_taste, dim=1)

        all_preds.extend(predictions_lbl.cpu().numpy() + predictions_taste.cpu().numpy()) 
        all_labels.extend(labels.cpu().numpy())

        correct_lbl += (predictions_lbl == labels).sum().item()
        correct_taste += (predictions_taste == tastes).sum().item()
        total += labels.size(0)

    accuracy_lbl = correct_lbl / total
    accuracy_taste = correct_taste / total
    print(f"Incremental Test Accuracy on Task {task_num}: {accuracy_lbl:.4f} | {accuracy_taste:.4f}")

    return accuracy_lbl, accuracy_taste

# ✅ Compute Anytime Average Accuracy
task_accuracies = {}

def compute_anytime_accuracy():
    avg_accuracies_lbl = []
    avg_accuracies_taste = []
    sorted_tasks = sorted(task_accuracies.keys())  # Ensure task order (e.g., [5, 10, 20])

    for i in range(len(sorted_tasks)):
        current_tasks = sorted_tasks[: i + 1]  # Get all tasks up to the current one
        avg_acc_lbl = np.mean([task_accuracies[t][-1][0] for t in current_tasks if task_accuracies[t]])
        avg_acc_taste = np.mean([task_accuracies[t][-1][1] for t in current_tasks if task_accuracies[t]])
        avg_accuracies_lbl.append(avg_acc_lbl)
        avg_accuracies_taste.append(avg_acc_taste)

    print("\n📊 Anytime Average Accuracies:")
    for t, acc_lbl, acc_taste in zip(sorted_tasks, avg_accuracies_lbl, avg_accuracies_taste):
        print(f"🟢 After {t} tasks: {acc_lbl:.4f}, {acc_taste:.4f}")

    return avg_accuracies_lbl, avg_accuracies_taste

# ✅ Compute Forgetting Measure
def compute_forgetting():
    forget_scores_lbl, forget_scores_taste = [], []
    for label, acc_list in task_accuracies.items():
        max_acc_lbl = max(acc[0] for acc in acc_list)
        last_acc_lbl = acc_list[-1][0]
        forget_scores_lbl.append(max_acc_lbl - last_acc_lbl)

    forgetting_measure_lbl = np.mean(forget_scores_lbl)
    print(f"\n🔻 Forgetting Measure (FM) - Label: {forgetting_measure_lbl:.4f}")

    for tastes, acc_list in task_accuracies.items():
        max_acc_taste = max(acc[1] for acc in acc_list)
        last_acc_taste = acc_list[-1][1]
        forget_scores_taste.append(max_acc_taste - last_acc_taste)

    forgetting_measure_taste = np.mean(forget_scores_taste)
    print(f"\n🔻 Forgetting Measure (FM) - Taste: {forgetting_measure_taste:.4f}")

    return forgetting_measure_lbl, forgetting_measure_taste

# ✅ Load BBBP Dataset
df_bbbp = pd.read_csv(bitter_d)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize Feature Extractor and Trainable Classifier
feature_extractor = MoLFormerFeatureExtractor().to(device)
classifier = MLPClassifier(input_dim=768, num_labels=df_bbbp["Label"].nunique(), num_tastes=df_bbbp["Taste"].nunique()).to(device)

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
fm_bitter = compute_forgetting()
dataset_forgetting["Bitter"] = fm_bitter
print("\n✅ Bitter Incremental Learning with Anytime Accuracy Computation Complete!")


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
fm_tox = compute_forgetting()
dataset_forgetting["Tox21"] = fm_tox
print("\n✅ Tox Incremental Learning with Anytime Accuracy Computation Complete!")


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
    def __init__(self, input_dim, hidden_dim=256, num_labels=2, num_fda=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_labels)
        self.fc3 = nn.Linear(hidden_dim, num_fda)

    def forward(self, x):
        h          = self.relu(self.fc1(x))
        logits_lbl = self.fc2(h)
        logits_fda = self.fc3(h)
        return logits_lbl, logits_fda

# ✅ SMILES Dataset Loader
class SMILESDataset(Dataset):
    def __init__(self, dataframe):
        self.smiles = dataframe["SMILES"].values
        self.labels = dataframe["Label"].values
        self.fda = dataframe["FDA_APPROVED"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx], self.fda[idx]

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
        for smiles, labels, fda in self.dataloader:
            smiles, labels, fda = list(smiles), labels.to(self.device), fda.to(self.device)
            features = feature_extractor(smiles)  # Extract features
            out_lbl, out_fda = self.model(features)  # Forward pass

            loss = nn.CrossEntropyLoss()(out_lbl, labels) + nn.CrossEntropyLoss()(out_fda, fda)  # Compute loss
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

    for smiles, labels, fda in train_loader:
        smiles, labels, fda = list(smiles), labels.to(device), fda.to(device)
        features = model(smiles).to(device)

        optimizer.zero_grad()
        out_lbl, out_fda = classifier(features)
        loss = criterion(out_lbl, labels) + criterion(out_fda, fda)

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
    correct_lbl = 0
    correct_fda = 0
    total = 0
    all_preds = []
    all_labels = []

    for smiles, labels, fda in test_loader:
        smiles, labels, fda = list(smiles), labels.to(device), fda.to(device)
        features = model(smiles).to(device)
        out_lbl, out_fda = classifier(features)
        predictions_lbl = torch.argmax(out_lbl, dim=1)
        predictions_fda = torch.argmax(out_fda, dim=1)

        all_preds.extend(predictions_lbl.cpu().numpy() + predictions_fda.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct_lbl += (predictions_lbl == labels).sum().item()
        correct_fda += (predictions_fda == fda).sum().item()
        total += labels.size(0)

    accuracy_lbl = correct_lbl / total
    accuracy_fda = correct_fda / total
    print(f"Incremental Test Accuracy on Task {task_num}: {accuracy_lbl:.4f}")
    print(f"Incremental Test FDA Accuracy on Task {task_num}: {accuracy_fda:.4f}")

    return accuracy_lbl, accuracy_fda

# ✅ Compute Anytime Average Accuracy
task_accuracies = {}

def compute_anytime_accuracy():
    avg_accuracies_lbl, avg_accuracies_fda = [], []
    sorted_tasks = sorted(task_accuracies.keys())  # Ensure task order (e.g., [5, 10, 20])

    for i in range(len(sorted_tasks)):
        current_tasks = sorted_tasks[: i + 1]  # Get all tasks up to the current one
        avg_acc_lbl = np.mean([task_accuracies[t][-1][0] for t in current_tasks if task_accuracies[t]])
        avg_accuracies_lbl.append(avg_acc_lbl)
        avg_acc_fda = np.mean([task_accuracies[t][-1][1] for t in current_tasks if task_accuracies[t]])
        avg_accuracies_fda.append(avg_acc_fda)

    print("\n📊 Anytime Average Accuracies:")
    for t, l,f in zip(sorted_tasks, avg_accuracies_lbl, avg_accuracies_fda):
        print(f"🟢 After {t} tasks: {l:.4f}")
        print(f"🟢 After {t} tasks: {f:.4f}")

    return avg_accuracies_lbl, avg_accuracies_fda

# ✅ Compute Forgetting Measure
def compute_forgetting():
    forget_scores_lbl, forget_scores_fda = [], []
    for label, acc_list in task_accuracies.items():
        max_acc_lbl = max(acc[0] for acc in acc_list)
        last_acc_lbl = acc_list[-1][0]
        forget_scores_lbl.append(max_acc_lbl - last_acc_lbl)

    forgetting_measure_lbl = np.mean(forget_scores_lbl)
    print(f"\n🔻 Forgetting Measure (FM) - Label: {forgetting_measure_lbl:.4f}")

    for fda, acc_list in task_accuracies.items():
        max_acc_fda = max(acc[1] for acc in acc_list)
        last_acc_fda = acc_list[-1][1]
        forget_scores_fda.append(max_acc_fda - last_acc_fda)

    forgetting_measure_fda = np.mean(forget_scores_fda)
    print(f"\n🔻 Forgetting Measure (FM) - FDA: {forgetting_measure_fda:.4f}")

    return forgetting_measure_lbl, forgetting_measure_fda

# ✅ Load BBBP Dataset
df_bbbp = pd.read_csv(clintox_d)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize Feature Extractor and Trainable Classifier
feature_extractor = MoLFormerFeatureExtractor().to(device)
classifier = MLPClassifier(input_dim=768, num_labels=df_bbbp["Label"].nunique(), num_fda=df_bbbp["FDA_APPROVED"].nunique()).to(device)

# ✅ EWC Setup
ewc = EWC(classifier, None, device)

# ✅ Incremental Learning

for task in incremental_tasks:
    print(f"\n🚀 Incremental Learning on {task} Classes")

    selected_data = df_bbbp
    train_data = SMILESDataset(selected_data)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    ewc.dataloader = train_loader  # Update EWC dataloader
    train_incremental(feature_extractor, classifier, train_loader, ewc, device)

    acc_lbl, acc_fda = evaluate_incremental(feature_extractor, classifier, train_loader, device, task)

    if task not in task_accuracies:
        task_accuracies[task] = []
    task_accuracies[task].append((acc_lbl, acc_fda))

# ✅ Compute Anytime Average Accuracy
compute_anytime_accuracy()

# ✅ Compute Forgetting Measure
fm_clintox = compute_forgetting()
dataset_forgetting["Clintox"] = fm_clintox
print("\n✅ Clintox Incremental Learning with Anytime Accuracy Computation Complete!")


# In[ ]:

# ─── API integration ─────────────────────────────────────────────────────────────

# will point at the final `feature_extractor` & `classifier` from your loops
api_feature_extractor = None
api_classifier        = None

def initialize_pipeline():
    """
    Call this once at FastAPI startup.  It hooks into the
    globals you trained at the bottom of this file.
    """
    global api_feature_extractor, api_classifier
    # assume `feature_extractor` & `classifier` were last set by your final EWC loop
    if api_feature_extractor is None or api_classifier is None:
        api_feature_extractor = feature_extractor
        api_classifier        = classifier
        api_classifier.eval()
    return api_feature_extractor, api_classifier

def predict_properties(records: list[str]) -> list[dict]:
    """
    Run inference on a list of SMILES.  Returns:
      [{"property":"predicted_label","value":int or None,"confidence":float}, …]
    """
    fe, cls = initialize_pipeline()
    out_recs = []
    for smi in records:
        # start with original metadata
        out = {"smiles": smi}

        if not smi.strip():
            # empty SMILES → all preds None/0
            for head in ("label","taste","fda","tox21_id"):
                out[f"predicted_{head}"]  = None
                out[f"confidence_{head}"] = 0.0
            out_recs.append(out)
            continue

        feats = fe([smi])
        with torch.no_grad():
            logits = cls(feats)

        # normalize to a dict of head→logits
        if isinstance(logits, tuple):
            # your two‐head model prints (label, taste) or (label, fda)
            # assume order matches these names; adjust if needed
            # here we bind first to "label" and second to whichever meta the rec had
            second_head = "fda" if "FDA_APPROVED" in smi else "taste"
            out_logits = {"label": logits[0], second_head: logits[1]}
        elif isinstance(logits, dict):
            out_logits = logits
        else:
            # single head only
            out_logits = {"label": logits}

        # now for each of the four possible heads do softmax & argmax
        for head in ("label","taste","fda","tox21_id"):
            if head in out_logits:
                probs = torch.softmax(out_logits[head], dim=1)[0].cpu().numpy()
                idx   = int(probs.argmax())
                conf  = float(probs[idx])
                out[f"predicted_{head}"]  = idx if conf>0.7 else None
                out[f"confidence_{head}"] = conf
            else:
                out[f"predicted_{head}"]  = None
                out[f"confidence_{head}"] = 0.0

        out_recs.append(out)
    return out_recs

def get_forgetting_measures() -> Dict[str, Any]:
    """
    Returns a dict mapping dataset name → its computed forgetting measure.
    """
    return dataset_forgetting
# ────────────────────────────────────────────────────────────────────────────────