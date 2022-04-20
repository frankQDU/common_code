import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn as nn


def cal_score(pred, label):
    pred = np.array(pred)
    label = np.array(label)

    diff = (pred - label) / 7
    diff = np.abs(diff)

    score = 100 * (1 - np.mean(diff))
    return score


class AQYDataset(Dataset):
    def __init__(self, df, device):
        self.user_id_list = df['user_id'].values

        self.launch_seq_list = df['launch_seq'].values

        self.label_list = df['label'].values

    def __getitem__(self, index):
        user_id = self.user_id_list[index]

        launch_seq = np.array(self.launch_seq_list[index])

        label = self.label_list[index]

        return user_id, launch_seq, label

    def __len__(self):
        return len(self.user_id_list)


def fit(model, train_loader, optimizer, criterion, device):
    model.train()

    pred_list = []
    label_list = []

    for user_id, launch_seq, label in tqdm(train_loader):
        user_id = user_id.long().to(device)
        launch_seq = launch_seq.long().to(device)
        label = torch.tensor(label).float().to(device)

        pred = model(user_id, launch_seq)

        loss = criterion(pred.squeeze(), label)
        loss.backward()
        optimizer.step()
        model.zero_grad()

        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = cal_score(pred_list, label_list)

    return score


def validate(model, val_loader, device):
    model.eval()

    pred_list = []
    label_list = []

    for user_id, launch_seq, label in tqdm(val_loader):
        user_id = user_id.long().to(device)
        launch_seq = launch_seq.long().to(device)
        label = torch.tensor(label).float().to(device)

        pred = model(user_id, launch_seq)

        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = cal_score(pred_list, label_list)

    return score


def predict(model, test_loader, device):
    model.eval()
    test_pred = []
    for user_id, launch_seq, _ in tqdm(test_loader):
        user_id = user_id.long().to(device)
        launch_seq = launch_seq.long().to(device)

        pred = model(user_id, launch_seq).squeeze()
        test_pred.extend(pred.cpu().detach().numpy())

    return test_pred


class AQYModel(nn.Module):
    def __init__(self):
        super(AQYModel, self).__init__()

        self.user_id_embedding = nn.Embedding(600000 + 1, 16)
        self.launch_type_embedding = nn.Embedding(2 + 1, 16)

        self.launch_seq_gru = nn.GRU(input_size=16,
                                     hidden_size=16,
                                     batch_first=True)

        self.fc = nn.Linear(32, 1)

    def forward(self, user_id, launch_seq):
        user_id_emb = self.user_id_embedding(user_id)

        # launch_seq = launch_seq.reshape((-1, 32, 1))
        launch_seq = self.launch_type_embedding(launch_seq)

        launch_seq, _ = self.launch_seq_gru(launch_seq)
        launch_seq = torch.mean(launch_seq, dim=1)

        fc_input = torch.cat([user_id_emb, launch_seq], 1)

        pred = self.fc(fc_input)

        return pred
