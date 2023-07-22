import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
from torch import nn
from tqdm import tqdm
import pandas

# set a random seed for reproducibility
myseed = 6666  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files=None):
        super().__init__()
        self.path = path
        if files != None:
            self.files = files
        else:
            self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        print(f"One {path} sample", self.files[0])
        self.transform = tfm
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_name = self.files[index]
        image = Image.open(file_name)
        image = self.transform(image)
        try:
            label = int(file_name.split("\\")[-1].split("_")[0])
        except:
            label = -1
        return image, label
    

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),# batch * [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), #[128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1) # batchsize * 维度
        return self.fc(out)


# 超参数
_exp_name = "sample"
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
n_epochs = 3
patience = 300 # if no improvement in 'patience' count, early stop
learning_rate = 0.0003
#数据
train_set = FoodDataset("training", tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = FoodDataset("validation", tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_set = FoodDataset("test", tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

#模型
model = Classifier().to(device)
#损失函数
criterion = nn.CrossEntropyLoss()
#优化算法
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

if __name__ == "__main__":
    model.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
    model.eval()
    prediction = []
    with torch.no_grad():
        for data, _ in tqdm(test_loader):
            test_pred = model(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.tolist()
    
    def pad4(i):
        return '0' * (4 - len(str(i))) + str(i)
    df = pandas.DataFrame()
    df["Id"] = [pad4(i) for i in range(1, len(test_set) + 1)]
    df["Categories"] = prediction
    df.to_csv("submission.csv", index=False)
    # train
    # stale = 0
    # best_acc = 0
    # for epoch in range(n_epochs):
    #     model.train()
    #     train_loss = []
    #     train_acc = []
    #     for imgs, labels in tqdm(train_loader):
    #         logits = model(imgs.to(device))
    #         loss = criterion(logits, labels.to(device))
    #         optimizer.zero_grad()
    #         loss.backward()
    #         grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=0)
    #         optimizer.step()

    #         acc = (logits.argmax(dim=1) == labels.to(device)).float().mean()

    #         train_loss.append(loss.item())
    #         train_acc.append(acc)
    #     train_loss = sum(train_loss) / len(train_loss)
    #     train_acc = sum(train_acc) / len(train_acc)

    #     # Print the information.
    #     print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    #     model.eval()
    #     valid_loss = []
    #     valid_acc = []
    #     for imgs, labels in tqdm(valid_loader):
    #         with torch.no_grad():
    #             logits = model(imgs.to(device))
    #         loss = criterion(logits, labels.to(device))

    #         acc = (logits.argmax(dim=1) == labels.to(device)).float().mean()

    #         valid_loss.append(loss.item())
    #         valid_acc.append(acc)
    #     valid_loss = sum(valid_loss) / len(valid_loss)
    #     valid_acc = sum(valid_acc) / len(valid_acc)

    #     # Print the information.
    #     print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    #     if valid_acc > best_acc:
    #         with open(f"./{_exp_name}_log.txt","a"):
    #             print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    #     else:
    #         with open(f"./{_exp_name}_log.txt","a"):
    #             print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    #     if valid_acc > best_acc:
    #         print(f"Best model found at epoch {epoch}, saving model")
    #         torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
    #         best_acc = valid_acc
    #         stale = 0
    #     else:
    #         stale += 1
    #         if stale > patience:
    #             print(f"No improvment {patience} consecutive epochs, early stopping")
    #             break