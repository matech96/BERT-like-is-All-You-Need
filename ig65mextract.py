# -*- coding: utf-8 -*-
# %%
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.transforms import Compose

import numpy as np
from einops.layers.torch import Rearrange, Reduce
from tqdm import tqdm

from ig65m.models import r2plus1d_34_32_ig65m
from ig65m.datasets import VideoDataset
from ig65m.transforms import ToTensor, Resize, Normalize


class VideoModel(nn.Module):
    def __init__(self, pool_spatial="mean", pool_temporal="mean"):
        super().__init__()

        self.model = r2plus1d_34_32_ig65m(num_classes=359, pretrained=True, progress=True)#.to('cuda:2')

        self.pool_spatial = Reduce("n c t h w -> n c t", reduction=pool_spatial)
        self.pool_temporal = Reduce("n c t -> n c", reduction=pool_temporal)
        

    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.pool_spatial(x)
        x = self.pool_temporal(x)

        return x

class IG65MExtract:
    def __init__(self, frame_size = 112, pool_spatial = 'mean', pool_temporal = 'mean'):
        if torch.cuda.is_available():
            print("🐎 Running on GPU(s)", file=sys.stderr)
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            print("🐌 Running on CPU(s)", file=sys.stderr)
            self.device = torch.device("cpu")

        self.model = VideoModel(pool_spatial=pool_spatial,
                           pool_temporal=pool_temporal)

        self.model.eval()

        for params in self.model.parameters():
            params.requires_grad = False

        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model)

        self.transform = Compose([
            ToTensor(),
            Rearrange("t h w c -> c t h w"),
            Resize(frame_size),
            Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])

    # dataset = WebcamDataset(clip=32, transform=transform)

    def predict(self, video_path, features_path, batch_size = 1):
        dataset = VideoDataset(video_path, clip=32, transform=self.transform)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

        features = []

        with torch.no_grad():
            for inputs in loader: # , total=len(dataset) // batch_size
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                outputs = outputs.data.cpu().numpy()

                for output in outputs:
                    features.append(output)

        np.save(features_path, np.array(features), allow_pickle=False)


# %%

if __name__ == "__main__":
    extractor = IG65MExtract()
    data_dir = Path('mosi_data')
    for split in ["train", "valid", "test"]:
        faces_dir = data_dir / split / 'video'
        videos = list(faces_dir.glob('*.mp4'))

        for video_path in tqdm(videos):
            video_name = video_path.stem
            feature_path = faces_dir / f"{video_name}_ig65m.npy"
            extractor.predict(video_path, feature_path)
