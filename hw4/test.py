import json
from pathlib import Path
import sys
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import time

def collate_batch(batch):
	# Process features within a batch.
	"""Collate a batch of data."""
	mel, speaker = zip(*batch)
	# Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
	mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
	# mel: (batch size, length, 40)
	return mel, torch.FloatTensor(speaker).long()

# a = torch.randn((3, 3))
# b = torch.randn((5, 3))
# a1 = torch.tensor(1)
# b1 = torch.tensor(2)
# batch = ((a, a1), (b, b1))
# r1, r2 = zip(*batch)
# print(r1)
# r1 = pad_sequence(r1, batch_first=True, padding_value=-20)
# print(r1)
# print(collate_batch(batch))


# r3 = torch.tensor([[[1, 2], [3, 4]],
# 	      [[4, 5], [6, 7]],
# 		  [[7, 8], [9, 10]]]).float()
# print(r3.shape)
# r3 = r3.mean(dim=1)
# print(r3)

# for i in tqdm(range(100), desc="hello"):
# 	time.sleep(0.5)

dict = {"a":123,"b":456}
for i in tqdm(range(10),total=10,desc = "WSX",ncols = 0,postfix = dict,mininterval = 0.3):
    pass