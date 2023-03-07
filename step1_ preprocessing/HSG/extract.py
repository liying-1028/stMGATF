from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image

from train import Model


class MyDataset(Dataset):
    def __init__(
        self,
        transform,
    ):
        super().__init__()

        self.transform = transform
        self.paths = list(Path("images/1").glob("*.jpeg"))

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
        batch_size=256,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    model = Model.load_from_checkpoint(
        "lightning_logs/version_4/checkpoints/epoch=3-step=20020.ckpt"
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
            np.save(f"features/{file_name}.npy", feature.numpy())


if __name__ == "__main__":
    main()
