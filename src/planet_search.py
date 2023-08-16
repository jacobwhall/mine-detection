"""

Find and download imagery using the Planet API

Utilizes code from Planet Labs notebooks (see links in references below)


References:

API basics
https://developers.planet.com/quickstart/apis/
https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/data-api-tutorials/search_and_download_quickstart.ipynb

Bulk orders
https://developers.planet.com/docs/orders/
https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/orders/ordering_and_delivery.ipynb
https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/orders/tools_and_toolchains.ipynb

Result limits / rate limiting
https://developers.planet.com/docs/analytics/api-mechanics/
https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/analytics/quickstart/02_fetching_feed_results.ipynb

Items/assets/sensors:
https://planet.com/docs/reference/data-api/items-assets
https://developers.planet.com/docs/apis/data/sensors/

# -------------------------------------

To cite Planet data in publications, please use the following:

Planet Team (2018). Planet Application Program Interface: In Space for Life on Earth. San Francisco, CA. https://api.planet.com.

```
@Misc{,
  author =    {Planet Team},
  organization = {Planet},
  title =     {Planet Application Program Interface: In Space for Life on Earth},
  year =      {2018--},
  url = "https://api.planet.com"
}
```

"""

import asyncio
import datetime
import glob
import json
import os
import random
import time
from datetime import date, datetime
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
import planet
import requests
import tomli
from planet import Session, data_filter
from planet.order_request import build_request, product
from shapely.geometry import shape

from config import load_config

# project config (different from planet-specific config, though I'd like to merge them at some point)
config = load_config()

from planet_utils import (auth, existing_df_path, is_planet, is_sentinel,
                          project_config, save_progress_df)

progress_df_cols = [
    "id",
    "country",
    "search_request",
    "api_response",
    "download_paths",
]


# create a Pandas dataframe to hold the search and download progress
# if it already exists, load that
def init_df(existing_df_path=existing_df_path, use_existing=True) -> pd.DataFrame:
    # if we are resuming progress from a previous search
    # existing_fname is the path to a pickled selected_df from before
    if not use_existing:
        return_df = pd.DataFrame(columns=progress_df_cols)
        return_df.set_index("id", inplace=True)
        return return_df
    elif existing_df_path is None:
        raise Exception("use_existing is True but no existing_fname was specified.")
    elif existing_df_path.exists():
        print(f"reading existing dataframe from {existing_df_path}")
        existing_df = pd.read_pickle(existing_df_path)
        existing_df.set_index("id", inplace=True)
        return existing_df
    else:
        print(f"no existing dataframe found at {existing_df_path}, will create one...")
        return_df = pd.DataFrame(columns=progress_df_cols)
        return_df.set_index("id", inplace=True)
        return return_df


async def search(sfilter):
    async with Session(auth=auth) as sess:
        cl = sess.client("data")
        items = [i async for i in cl.search(sfilter["item_types"], sfilter["filter"])]
        return items


def to_datetime(input_dt=Union[date, datetime]) -> datetime:
    if isinstance(input_dt, datetime):
        return input_dt
    elif isinstance(input_dt, date):
        return datetime.combine(input_dt, datetime.min.time())
    else:
        return ValueError(
            "temporal_start and temporal_end need to be a date or datetime"
        )


def create_search_filters(bounding_box, minimum_percent_usable: int):
    # these are search filters to be passed to the Planet API
    filters = [
        data_filter.geometry_filter(bounding_box.__geo_interface__),
        data_filter.date_range_filter("acquired", gte=temporal_start, lt=temporal_end),
        data_filter.string_in_filter("instrument", project_config["instrument_list"]),
        data_filter.string_in_filter("quality_category", ["standard"]),
    ]

    # add filter to select scenes with limited clouds
    if is_planet:
        filters.extend(
            [
                data_filter.range_filter(
                    "cloud_percent", lte=100 - minimum_percent_usable
                ),
                # data_filter.range_filter('clear_confidence_percent', gte=50),
                data_filter.string_in_filter("publishing_stage", ["finalized"]),
            ]
        )
    elif is_sentinel:
        filters.extend(
            [
                data_filter.range_filter(
                    "usable_data", gt=minimum_percent_usable / 100.0
                ),
            ]
        )
    else:
        raise ValueError(f"item_type not supported ({item_type})")

    return data_filter.and_filter(filters)


if __name__ == "__main__":
    project_path = Path(project_config["path"])

    # start and end of time range to search for imagery
    temporal_start = to_datetime(project_config["temporal_start"])
    temporal_end = to_datetime(project_config["temporal_end"])

    # this is the Planet item type
    item_type = project_config["item_type"]

    update_missing = project_config["update_missing"]
    fix_coverage = project_config["fix_coverage"]

    bounds_path = project_path / config["mine_footprints_geojson_path"]
    bounds_gdf = gpd.read_file(bounds_path)

    # load progress_df from disk, or create it if it doesn't exist
    progress_df = init_df(
        existing_df_path=existing_df_path, use_existing=project_config["use_existing"]
    )

    # For each row in bounds_gdf, add it to progress_df if it isn't already there
    new_rows = []
    for ix, row in bounds_gdf.iterrows():
        if row["id"] in progress_df.index.values:
            # TODO: set selected column to True if any of these rows do not have an api_response
            print("there is already an entry for this footprint in progress_df")
            continue

        search_name = f"{row.id}: {str(temporal_start)[:10]}-{str(temporal_end)[:10]}  [{random.randint(0, 1e6)}]"
        print(f"searching for {search_name}")

        # generate search filter with footprint bounding box and cloud cover limit
        # NOTE: it's important that we calculate the bounding box here. row["geometry"] is the full geometry of the feature.
        combined_filter = create_search_filters(
            row["geometry"].envelope, int(project_config["minimum_percent_usable"])
        )

        # API request object
        search_request = {
            "name": search_name,
            "item_types": [item_type],
            "filter": combined_filter,
        }

        new_rows.append(
            {
                "id": row["id"],
                "country": row["COUNTRY_NAME"],
                "search_request": [search_request],
                "selected": True,
            }
        )

    if len(new_rows) == 0:
        print("nothing new to search for, quitting...")
        exit()

    # dataframe just for the rows we are adding in this search
    this_progress_df = pd.DataFrame(data=new_rows, columns=progress_df_cols)

    print("running search queries through API...")
    breakpoint()

    # query API with search request for each row
    def _search(req):
        if req is not None:
            time.sleep(0.5)
            print("working...")
            return asyncio.run(search(req[0]))

    this_progress_df["api_response"] = this_progress_df["search_request"].apply(_search)

    # add this_progress_df to progress_df
    progress_df = pd.concat([progress_df, this_progress_df])

    # save progress dataframe to disk before continuing with downloads
    if project_config["use_existing"]:
        print("saving search results to file...")
        save_progress_df(progress_df)

    # TODO: figure out if the items have already been downloaded?
    """
    all_item_ids = list(set(match_df.selected.to_list()))

    local_item_list = [
        os.path.basename(i).replace(end_str, "")
        for end_str in asset_file_strings
        for i in glob.glob(
            str(project_path / "data" / "imagery" / "*" / item_type / f"*{end_str}")
        )
    ]

    item_ids = [i for i in all_item_ids if i not in local_item_list]
    """
