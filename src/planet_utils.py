from pathlib import Path

import pandas as pd
import tomli
from planet import Auth

config_path = Path("planet.toml")

# load configuration file
# The needs to be done first because some values are used as function defaults
try:
    with open(config_path, "rb") as config_file:
        config = tomli.load(config_file)
except FileNotFoundError:
    raise FileNotFoundError(
        "planet.toml not found, are you running this from the root of the repo?"
    )

project = config["main"]["project"]
project_config = config[project]
existing_df_path = Path(project_config["existing_fname"])

item_type = project_config["item_type"]

is_planet = item_type.lower() in [
    "psscene",
    "psscene3band",
    "psscene4band",
    "psorthotile",
    "skysatscene",
    "skysatcollect",
    "skysatvideo",
]
is_sentinel = item_type in ["sentinel2l1c"]


# create Planet authentication object from key
with open(config["login"]["api_key_file"], "r") as api_key_file:
    auth = Auth.from_key(api_key_file.read().strip())


# save progress dataframe to disk
def save_progress_df(progress_df: pd.DataFrame, dst=existing_df_path):
    progress_df.to_pickle(dst)
