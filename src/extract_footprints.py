from pathlib import Path

import geopandas as gpd

from config import load_config

config = load_config()
mine_footprints_path = Path(config["project_path"]) / config["mine_footprints_path"]
mine_footprints_geojson_path = (
    Path(config["project_path"]) / config["mine_footprints_geojson_path"]
)


def generate_envelopes(dst=mine_footprints_geojson_path, overwrite=True):
    if overwrite or not Path(dst).exists():
        print("Loading mine polygons")
        data = gpd.read_file(mine_footprints_path)
        # geopandas will treat the fid as an index column, name it as "id" so it will show up in geojson export
        data.index.rename("id", inplace=True)

        if len(config["countries"]) > 0:
            print(
                "Filtering polygons to within {}".format(", ".join(config["countries"]))
            )
            data = data.loc[data["COUNTRY_NAME"].isin(config["countries"])].copy()

        # print("Calculating polygon boundary boxes")
        # data.geometry = data.envelope

        print(f"Saving polygons to {dst}")
        data.to_file(dst, index=True, crs="EPSG:4326")


if __name__ == "__main__":
    generate_envelopes()
