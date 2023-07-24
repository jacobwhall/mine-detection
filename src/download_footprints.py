import os
import urllib.request
from pathlib import Path

from config import load_config

config = load_config()
mine_footprints_path = Path(config["project_path"]) / config["mine_footprints_path"]


def download_footprints(
    url="https://download.pangaea.de/dataset/942325/files/global_mining_polygons_v2.gpkg",
    dst=mine_footprints_path,
    overwrite=True,
):
    if dst.exists() and not overwrite:
        print(
            "Skipping mine footprints download, file already exists and overwrite set to False."
        )
    else:
        print(f"Downloading mine footprints to {dst}...")
        os.makedirs(dst.parent, exist_ok=True)
        urllib.request.urlretrieve(url, dst)


if __name__ == "__main__":
    download_footprints()
