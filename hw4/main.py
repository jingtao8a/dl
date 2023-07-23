import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import json
import os
from torch.nn.utils.rnn import pad_sequence
from classifier import Classifier
from learningratewarmup import get_cosine_schedule_with_warmup
from torch import nn
from tqdm import tqdm

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(87)


class MyDataset(Dataset):
    def __init__(self, data_dir, segment_len = 128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(open(mapping_path))
        self.speaker2id = mapping["speaker2id"]

        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(open(metadata_path))["speakers"]

        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel) - self.segment_len)
            mel = torch.FloatTensor(mel[start:start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)

        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker
    
    def get_speaker_number(self):
        return self.speaker_num
    
def collate_batch(batch):
	# Process features within a batch.
	"""Collate a batch of data."""
	mel, speaker = zip(*batch)
	# Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
	mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
	# mel: (batch size, length, 40)
	return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size):
    dataset = MyDataset(data_dir)
    speak_num = dataset.get_speaker_number()
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,pin_memory=True, collate_fn=collate_batch)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=True, collate_fn=collate_batch)

    return train_loader, valid_loader, speak_num

def model_fn(batch, model, criterion, device):
    mels, labels = batch
    mels = mels.to(device)
    labels = labels.to(device)
    outs = model(mels)
    loss = criterion(outs, labels)
    preds = outs.argmax(1)
    accuracy = torch.mean((preds == labels).float())
    return loss, accuracy

def valid(dataloader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total = len(dataloader.dataset), ncols=100, desc="Valid", unit="uttr")
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()
        
        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i + 1):.2f}",
            accuracy=f"{running_accuracy / (i + 1):.2f}"
        )
    pbar.close()
    model.train()

    return running_accuracy / len(dataloader)

def parse_args():
	config = {
		"data_dir": "./Dataset",
		"save_path": "model.ckpt",
		"batch_size": 32,
		"valid_steps": 2000,
		"warmup_steps": 1000,
		"save_steps": 10000,
		"total_steps": 70000,
	}

	return config

def main(data_dir,
	save_path,
	batch_size,
	valid_steps,
	warmup_steps,
	total_steps,
	save_steps,
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")
    #数据
    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!",flush = True)
    #模型
    model = Classifier(n_spks=speaker_num).to(device)
    #损失函数
    criterion = nn.CrossEntropyLoss()
    #优化算法
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps) #????
    print(f"[Info]: Finish creating model!",flush = True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=100, desc="Train", unit=" step")

    for step in range(total_steps):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.update()
        pbar.set_postfix(loss=f"{batch_loss:.2f}", accuracy=f"{batch_accuracy:.2f}", step=step+1)

        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_accuracy = valid(valid_loader, model, criterion, device)
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()
            
            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved, accuracy = {best_accuracy:.4f}")
    pbar.close()


if __name__ == "__main__":
    main(**parse_args())