import torch
import os
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import gc

def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)

def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # must be odd
    if concat_n < 2: # 2 * k + 1
        return x

    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = concat_n // 2
    for r_idx in range(1, mid + 1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41
    mode = 'train' if (split == 'train' or split == 'valid') else 'test'
    label_dict = {}
    if mode != 'test':
        phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

        for line in phone_file:
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'valid':
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \' split \' argument for dataset')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - #phone classes: ' + str(class_num) + ', number of utterances for ' + split + ':' + str(len(usage_list)))
    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
            label = torch.LongTensor(label_dict[fname])
        
        X[idx: idx + cur_len, :] = feat

        if mode != 'test':
            y[idx: idx + cur_len] = label
        
        idx += cur_len
    X = X[:idx, :]
    if mode != 'test':
        y = y[:idx]
    print(f'[INFO] {split} set')
    if mode != 'test':
        print(X.shape)
        return X, y
    else:
        return X
    
class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None
    
    def __getitem__(self, index):
        if self.label is not None:
            return self.data[index], self.label[index]
        else:
            return self.data[index]
        
    def __len__(self):
        return len(self.data)

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.block(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

concat_nframes = 1
train_ratio = 0.8
seed = 0
batch_size = 512
num_epoch = 5
learning_rate = 0.0001
model_path = './model.ckpt'

input_dim = 39 * concat_nframes
output_dim = 41
hidden_layers = 1
hidden_dim = 256
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

if __name__ == "__main__":
    # train_X, train_y = preprocess_data("train", "libriphone/feat", "libriphone", concat_nframes, train_ratio, seed)
    # val_X, val_y = preprocess_data("valid", "libriphone/feat", "libriphone", concat_nframes, train_ratio, seed)

    # train_set = LibriDataset(train_X, train_y)
    # val_set = LibriDataset(val_X, val_y)

    # del train_X, train_y, val_X, val_y
    # gc.collect()

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    same_seeds(seed)
    model = Classifier(input_dim, output_dim, hidden_layers, hidden_dim).to(device)
    
    # test
    model.load_state_dict(torch.load(model_path))
    test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes)
    test_set = LibriDataset(test_X, None)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model.eval()
    test_acc = 0.0
    test_lengths = 0
    pred = np.array([], dtype=np.int32)
    for i, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            features = batch
            features = features.to(device)
            output = model(features)
            _, test_pred = torch.max(output, 1)
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)
    with open('prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # train
    # best_acc = 0.0
    # for epoch in range(num_epoch):
    #     train_acc = 0.0
    #     train_loss = 0.0
    #     val_acc = 0.0
    #     val_loss = 0.0

    #     model.train()
    #     for i, batch in enumerate(tqdm(train_loader)):
    #         features, labels = batch
    #         features = features.to(device)
    #         labels = labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(features)

    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         _, train_pred = torch.max(outputs, 1)
    #         train_acc += (train_pred.detach() == labels.detach()).sum().item()
    #         train_loss += loss.item()

    #     #validation
    #     if len(val_set) > 0:
    #         model.eval()
    #         with torch.no_grad():
    #             for i, batch in enumerate(tqdm(val_loader)):
    #                 features, labels = batch
    #                 features = features.to(device)
    #                 labels = labels.to(device)
    #                 outputs = model(features)
    #                 loss = criterion(outputs, labels) 
                
    #                 _, val_pred = torch.max(outputs, 1) 
    #                 val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
    #                 val_loss += loss.item()
    #         print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
    #             epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
    #         ))

    #         # if the model improves, save a checkpoint at this epoch
    #         if val_acc > best_acc:
    #             best_acc = val_acc
    #             torch.save(model.state_dict(), model_path)
    #             print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
        
    #     else:
    #         print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
    #             epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
    #             ))
    
    # if len(val_set) == 0:
    #     torch.save(model.state_dict(), model_path)
    #     print('saving model at last epoch')

    # del train_loader, val_loader
    # gc.collect()
