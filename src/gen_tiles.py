import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import raster_geometry_mask
from rasterio.windows import Window
from tqdm import tqdm

from config import load_config

config = load_config()
mine_footprints_geojson_path = (
    Path(config["project_path"]) / config["mine_footprints_geojson_path"]
)
mine_footprints = gpd.read_file(mine_footprints_geojson_path)


def gen_tiles_for_image(image_path, mask_path, tile_dir, overwrite=True):
    # create directory for this image's raster features
    feature_dir = tile_dir / image_path.stem / "features"
    os.makedirs(feature_dir, exist_ok=True)

    # create directory for this image's raster and vector labels
    label_dir = tile_dir / image_path.stem / "labels"
    os.makedirs(label_dir, exist_ok=True)

    with rasterio.open(image_path) as image_src, rasterio.open(mask_path) as label_src:
        width, height = image_src.meta["width"], image_src.meta["height"]
        col_off = 0
        while col_off <= width - config["tile_dimension"]:
            row_off = 0
            while row_off <= height - config["tile_dimension"]:
                # this is the window we will use to read both the image and mask
                window = Window(
                    col_off, row_off, config["tile_dimension"], config["tile_dimension"]
                )

                # read clear mask band from udm2
                clear = label_src.read(
                    label_src.descriptions.index("clear") + 1, window=window
                )

                # read light_haze mask band from udm2
                light_haze = label_src.read(
                    label_src.descriptions.index("haze_light") + 1, window=window
                )

                # ndarray of same size as clear and light_haze, with True wherever it is clear or lightly hazy
                usable_mask = np.logical_or(clear, light_haze)

                # are there enough usable pixels to pass our threshold?
                if (usable_mask.sum() / usable_mask.size) > (
                    config["tile_percent_usable"] / 100
                ):
                    tile_feature_path = feature_dir / f"{col_off}_{row_off}.tif"
                    tile_label_path = label_dir / f"{col_off}_{row_off}.tif"

                    # if this tile has not been (completely) created before
                    # or overwrite=True
                    if (
                        overwrite
                        or not tile_feature_path.exists()
                        or not tile_label_path.exists()
                    ):
                        # just to be extra-sure we read bands in order of red, green, blue
                        band_indices = [
                            image_src.indexes[image_src.descriptions.index(band_name)]
                            for band_name in ["red", "green", "blue"]
                        ]

                        # adjust rasterio write settings for raster feature
                        write_kwargs = image_src.meta.copy()
                        write_kwargs.update(
                            {
                                "width": config["tile_dimension"],
                                "height": config["tile_dimension"],
                                "transform": rasterio.windows.transform(
                                    window, image_src.transform
                                ),
                                "count": 3,  # RGB
                                "dtype": "int16",
                            }
                        )

                        # write feature raster to tile_feature_path
                        with rasterio.open(
                            tile_feature_path,
                            "w",
                            **write_kwargs,
                        ) as dst:
                            dst.write(
                                image_src.read(
                                    band_indices,
                                    window=window,
                                ),
                            )
                            # label bands as red, green, blue
                            dst.set_band_description(1, "red")
                            dst.set_band_description(2, "green")
                            dst.set_band_description(3, "blue")

                        # now, let's open that file we just wrote in read mode and use its metadata to prepare a raster label (mask)
                        with rasterio.open(
                            tile_feature_path,
                            "r",
                            **write_kwargs,
                        ) as feature_src:
                            write_kwargs = feature_src.meta.copy()
                            write_kwargs.update(
                                {
                                    "count": 1,  # binary mask
                                }
                            )

                            # reproject mine_footprints to use the CRS of the raster feature
                            reprojected_mine_footprints = mine_footprints.to_crs(
                                feature_src.crs
                            )

                            # TODO: create a geodataframe with just the features that overlap this tile
                            clipped_mine_footprints = reprojected_mine_footprints.clip(
                                feature_src.bounds
                            )

                            # TODO: save that geodataframe to geojson vector label
                            if len(clipped_mine_footprints) == 0:
                                label_mask = np.zeros(
                                    (config["tile_dimension"], config["tile_dimension"])
                                )
                            else:
                                # generate mask using rasterio's mask function and the reprojected footprints
                                label_mask, _, _ = raster_geometry_mask(
                                    feature_src, clipped_mine_footprints.geometry
                                )

                            # now its time to write the raster label
                            with rasterio.open(
                                tile_label_path,
                                "w",
                                **write_kwargs,
                            ) as dst:
                                dst.write(np.array([label_mask]))

                # increment row_off
                row_off += config["tile_stride"]
            # increment col_off
            col_off += config["tile_stride"]


if __name__ == "__main__":
    config = load_config()
    mine_footprints_path = Path(config["project_path"]) / config["mine_footprints_path"]

    # TODO: read order ID out of pickled download logs
    # pickle_path = "data/download_progress.pickle"  # TODO: set this in config.toml, currently in planet.toml

    imagery_folder = (
        Path(config["project_path"]) / "data/raw" / config["order_id"] / "PSScene"
    )

    tile_dir = Path(config["project_path"]) / config["tile_dir"]

    for analytics_path in tqdm(list(imagery_folder.glob("*_AnalyticMS_SR.tif"))):
        udm2_path = analytics_path.with_stem(
            analytics_path.stem.replace("_AnalyticMS_SR", "_udm2")
        )
        if udm2_path.exists():
            gen_tiles_for_image(analytics_path, udm2_path, tile_dir=tile_dir / "all")
        else:
            print(f"Warning: no mask found for image {analytics_path}")

    # data = pd.read_pickle(pickle_path)
    # apply gen_tiles_for_row to each row
    # data.loc[data.download_paths.isna()].T.apply(gen_tiles_for_row)
