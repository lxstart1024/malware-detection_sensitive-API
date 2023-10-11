import math
from collections import defaultdict
import torch
from nltk.corpus import sentence_polarity
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import csv
import os
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


class TransformerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def length_to_mask(lengths):
    max_len = torch.max(lengths)
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
    return mask


def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets


class Transformer_Classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_class, dim_feedforward=512, num_head=6, num_layers=1,
                 dropout=0.1, max_len=128, activation: str = "relu"):
        super(Transformer_Classifier, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_head, dim_feedforward, dropout, activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        inputs = torch.transpose(inputs, 0, 1)
        hidden_states = inputs
        attention_mask = length_to_mask(lengths) == False
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states = hidden_states[0, :, :]
        output = self.output(hidden_states)
        log_probs = torch.log_softmax(output, dim=1)
        return log_probs


def test_training_with_PMM(MM_path, PM_path):
    embedding_dim = 768
    hidden_dim = 768
    num_class = 2
    batch_size = 32
    num_epoch = 10

    MM_file_list = os.listdir(MM_path)
    MM_data = []
    for filename in MM_file_list:
        temp_file = []
        filepath = MM_path + "/" + filename
        with open(filepath, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                data_re = [float(i) for i in eval(row[0])]
                temp_file.append(data_re)
        MM_data.append(temp_file)

    PM_file_list = os.listdir(PM_path)
    PM_data = []
    for filename in PM_file_list:
        temp_file = []
        filepath = PM_path + "/" + filename
        with open(filepath, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                data_re = [float(i) for i in eval(row[0])]
                temp_file.append(data_re)
        PM_data.append(temp_file)

    neg_samples = []
    for i in range(len(MM_data)):
        if random.random() < 0.5:
            neg_MM_data = random.choice(MM_data)
            temp_neg = neg_MM_data + PM_data[i]
            label_neg = 0
            neg_samples.append((temp_neg, label_neg))

    input_data = []
    for i in range(len(MM_data)):
        temp_data = MM_data[i] + PM_data[i]
        label = 1
        input_data.append((temp_data, label))
    if len(neg_samples) != 0:
        train_data = input_data + neg_samples

    train_dataset = TransformerDataset(train_data)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    device = torch.device("cpu")
    model = Transformer_Classifier(embedding_dim, hidden_dim, num_class)
    model.to(device)

    nll_loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    for epoch in range(num_epoch):
        total_loss = 0.0
        for batch in tqdm(train_dataset_loader, desc=f"Training Epoch{epoch}"):
            inputs, lengths, targets = [x.to(device) for x in batch]
            log_probs = model(inputs, lengths)
            loss = nll_loss(log_probs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            torch.save(model.state_dict(), "./weights/model-{}.pt".format(epoch))
        print(f"Loss:{total_loss:.2f}")


def test_training_classifier_with_PMM_init(path):
    embedding_dim = 768
    hidden_dim = 768
    num_class = 2
    batch_size = 32
    num_epoch = 10

    filelist = os.listdir(path)
    train_data = []
    for filename in filelist:
        if "Virus" in filename:
            file_label = 1
        else:
            file_label = 0
        temp_file = []
        filepath = path + "/" + filename
        with open(filepath, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                data_re = [float(i) for i in eval(row[0])]
                temp_file.append(data_re)
        train_data.append((temp_file, file_label))
    train_data = train_data

    train_set, test_set = train_test_split(train_data, test_size=0.2, random_state=42)

    train_dataset = TransformerDataset(train_set)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    test_dataset = TransformerDataset(test_set)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    device = torch.device("cpu")
    model = Transformer_Classifier(embedding_dim, hidden_dim, num_class)
    state_dict = torch.load("./weights/model-9.pt")
    model.load_state_dict(state_dict)
    model.to(device)
    nll_loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    def evaluate(model, dataloader):
        model.eval()
        predicted_labels = []
        true_labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, lengths, targets = [x.to(device) for x in batch]
                log_probs = model(inputs, lengths)
                _, predicted = torch.max(log_probs, 1)
                predicted_labels.extend(predicted.cpu().tolist())
                true_labels.extend(targets.cpu().tolist())
        return true_labels, predicted_labels

    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(train_data_loader, desc=f"Training Epoch{epoch}"):
            inputs, lengths, targets = [x.to(device) for x in batch]
            log_probs = model(inputs, lengths)
            loss = nll_loss(log_probs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Evaluate the model
        true_labels, predicted_labels = evaluate(model, train_data_loader)
        print("true_labels:", true_labels)
        print("predicted_labels:", predicted_labels)
        # Calculate evaluation metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        print(f"Accuracy: {accuracy:.4f}, Precision:{precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        print(f"Loss: {total_loss:.2f}")

    print("test")
    true_labels, predicted_labels = evaluate(model, test_data_loader)
    print("true_labels:", true_labels)
    print("predicted_labels:", predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}, Precision:{precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


if __name__ == '__main__':
    path = "F:/fused_feature_set"
    test_training_classifier_with_PMM_init(path)