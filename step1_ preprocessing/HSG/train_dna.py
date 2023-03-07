import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from resnet import get_resnet, name_to_params


class ImageNet(pl.LightningDataModule):
    def __init__(
        self,
        train_batch_size: int = 128,
        val_batch_size: int = 128,
        num_workers: int = 16,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def train_dataloader(self):
        dataset = ImageFolder(
            root="data/dna/train",
            transform=self.transform,
        )

        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = ImageFolder(
            root="data/dna/val",
            transform=self.transform,
        )

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


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


def main():
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
        max_epochs=30,
    )
    model = Model(
        learning_rate=1e-3,
        max_epochs=30,
    )
    model.load_state_dict(torch.load(
        "lightning_logs/version_4/checkpoints/epoch=3-step=20020.ckpt",
        map_location="cpu",
    )["state_dict"])
    model.fc = nn.Linear(2048, 20)
    trainer.fit(
        datamodule=ImageNet(
            train_batch_size=256,
            val_batch_size=256,
            num_workers=16,
        ),
        model=model,
    )


if __name__ == "__main__":
    main()
