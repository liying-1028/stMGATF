from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from resnet import get_resnet, name_to_params


class Model(pl.LightningModule):
    def __init__(
            self,
            learning_rate: float = 1e-4,
            max_epochs: int = 30,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        pth_path = "r152_3x_sk1.pth"
        self.resnet, _ = get_resnet(*name_to_params(pth_path))
        # self.resnet.load_state_dict(torch.load(pth_path, map_location="cpu")["resnet"])

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.reduce = nn.Linear(6144, 2048)
        self.fc = nn.Linear(2048, 1000)

    def forward(self, x):
        with torch.no_grad():
            self.resnet.eval()
            x = self.resnet(x)
        x = self.reduce(x)
        x = F.leaky_relu(x)
        return self.fc(x)

    def training_step(self, batch, batch_idx: int):
        images, labels = batch

        logits = self.forward(images)
        preds = logits.detach().argmax(-1)

        loss = F.cross_entropy(logits, labels)

        acc = (preds == labels).sum().item() / logits.shape[0] * 100

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log(
            "train/acc", acc,
            on_step=True, on_epoch=True, prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx: int):
        images, labels = batch

        logits = self.forward(images)
        preds = logits.detach().argmax(-1)

        loss = F.cross_entropy(logits, labels)

        acc = (preds == labels).sum().item() / logits.shape[0] * 100

        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.log(
            "val/acc", acc,
            on_step=True, on_epoch=True, prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


name = '151669'
import os
# 为每一类创建新文件夹
dir = 'features_{}'.format(name)
if not os.path.exists(dir):
    os.mkdir(dir)


class MyDataset(Dataset):
    def __init__(
            self,
            transform,
    ):
        super().__init__()

        self.transform = transform
        # 文件夹路径拼接
        self.paths = list(Path("images/" + name + "/" + "tmp").glob("*.jpeg"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, path.name


def main():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataloader = DataLoader(
        MyDataset(transform),
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model = Model.load_from_checkpoint(
        "epoch=3-step=20020.ckpt",
    )
    model.eval()
    model.cuda()

    for images, file_names in tqdm(dataloader):
        images = images.to("cuda", non_blocking=True)

        with torch.no_grad():
            x = model.resnet(images)
            x = model.reduce(x)
            x = x.cpu()

        for feature, file_name in zip(x, file_names):
            np.save(f"features_{name}/{file_name}.npy", feature.numpy())


if __name__ == "__main__":
    main()
