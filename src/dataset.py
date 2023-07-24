from pathlib import Path

import numpy as np
import rasterio
from torch.utils.data import Dataset


class MiningDataset(Dataset):
    def __init__(self, tile_dir: Path, transform=None):
        self.transform = transform
        self.tiles = []
        for i in tile_dir.iterdir():
            self.tiles.append((i / "feature.tif", i / "label.tif"))
        print(f"Loaded {len(self.tiles)} tiles")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        image_path, mask_path = self.tiles[idx]
        with rasterio.open(image_path) as image_src:
            # ndarray with shape 3, 224, 224 (Bands, Height, Width)
            image = image_src.read().astype("float32")

            # we have to re-order because the pytorch ToTensor callable wants the "Count" (band) to be last
            # humorously, ToTensor will then mutate it back to this original shape...oh well.
            # ndarray with shape 224, 224, 3 (Height, Width, Bands)
            image = np.moveaxis(image, 0, 2)

        with rasterio.open(mask_path) as mask_src:
            label = mask_src.read().astype("float32")
        if self.transform:
            image = self.transform(image)
        return image, label
