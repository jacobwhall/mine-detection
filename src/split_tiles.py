import os
import random
import shutil
from pathlib import Path

import rasterio
from tqdm import tqdm

from config import load_config

config = load_config()

tile_dir = Path(config["project_path"]) / config["tile_dir"]

try:
    assert (
        config["percent_training"]
        + config["percent_validation"]
        + config["percent_test"]
        == 100
    )
except AssertionError:
    raise ValueError(
        "Percent training, validation, and test values do not add to 100 in config!"
    )

if config["only_tiles_with_mines"]:
    print(
        "warning: only selecting tiles with mines drastically slows down tile split operation."
    )

# find all the tiles
tiles = []
no_mine_count = 0
no_mine_limit = 250
for image_dir in tqdm(list((tile_dir / "all").iterdir())):
    for raster_feature in tqdm(list((image_dir / "features").glob("*.tif"))):
        raster_label = image_dir / "labels" / raster_feature.name
        if raster_label.exists():
            if config["only_tiles_with_mines"]:
                with rasterio.open(raster_label) as label:
                    has_mines = label.read().any()
                    if not has_mines:
                        if no_mine_count < no_mine_limit:
                            no_mine_count += 1
                        else:
                            continue
            tiles.append((raster_feature, raster_label))
        else:
            raise FileNotFoundError(
                f"No raster label found for feature {raster_feature}"
            )

# shuffle tiles randomly
random.shuffle(tiles)

# save the tile count before we start popping tiles
initial_tile_count = len(tiles)
print(f"{len(tiles)} tiles found")


def create_tile_symlink(feature_path, label_path, dst):
    os.makedirs(dst)
    os.symlink(feature_path, dst / "feature.tif")
    os.symlink(label_path, dst / "label.tif")


num_validation = int(initial_tile_count * (config["percent_validation"] / 100))
print(f"setting aside {num_validation} tiles for validation...")
validation_dir = tile_dir / "validate"
if validation_dir.exists():
    shutil.rmtree(validation_dir)
for i in range(num_validation):
    create_tile_symlink(*tiles.pop(), validation_dir / str(i))


num_test = int(initial_tile_count * (config["percent_test"] / 100))
print(f"setting aside {num_test} tiles for testing...")
test_dir = tile_dir / "test"
if test_dir.exists():
    shutil.rmtree(test_dir)
for i in range(num_test):
    create_tile_symlink(*tiles.pop(), test_dir / str(i))

print(f"the rest ({len(tiles)}) will be used for training...")
training_dir = tile_dir / "train"
if training_dir.exists():
    shutil.rmtree(training_dir)
for i, tile in enumerate(tiles):
    create_tile_symlink(*tile, training_dir / str(i))
