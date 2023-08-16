import asyncio
import random
from pathlib import Path

import pandas as pd
import planet
import requests
import tomli
from planet import OrdersClient, Session
from planet.order_request import build_request, product
from planet.reporting import StateBar

from config import load_config
from planet_utils import auth, existing_df_path, project_config


async def order(products):
    new_request = build_request(
        f"bulk_order [{random.randint(0, 1e5)}]",
        [
            product(
                products[0]["item_ids"],
                products[0]["product_bundle"],
                products[0]["item_type"],
            )
        ],
    )
    async with Session(auth=auth) as sess:
        cl = OrdersClient(sess)
        this_order = await cl.create_order(new_request)
        return this_order


async def wait_on_order(order_id):
    async with Session(auth=auth) as sess:
        cl = OrdersClient(sess)
        with StateBar() as bar:
            await cl.wait(order_id, callback=bar.update_state)


async def download_order(order_id, outputs_path):
    async with Session(auth=auth) as sess:
        cl = OrdersClient(sess)
        dl = await cl.download_order(
            order_id, directory=outputs_path, overwrite=False, progress_bar=True
        )


if __name__ == "__main__":
    if not existing_df_path.exists():
        raise FileNotFoundError("No existing dataframe found to read from!")
    else:
        progress_df = pd.read_pickle(existing_df_path)

    project_path = Path(project_config["path"])

    # not the planet-specific config file, this is the primary config
    config = load_config()
    refined_df = progress_df[progress_df["country"].isin(config["countries"])]

    item_ids = []
    refined_df["api_response"].apply(
        lambda r: item_ids.extend([i["id"] for i in r[: config["max_images"] - 1]])
    )

    outputs_path = project_path / "data/raw"
    outputs_path.mkdir(parents=True, exist_ok=True)

    # define products part of order
    # TODO: build this list with an entry for each product bundle in a list of product bundles
    products = [
        {
            "item_ids": [x["id"] for x in refined_df.api_response[0]],
            "item_type": project_config["item_type"],
            "product_bundle": project_config["product_bundle"],
        }
    ]

    # Submit order to Planet
    order = asyncio.run(order(products))
    order_id = order["id"]
    print(f"Submitted order (id:\n{order_id})")

    # Wait for order to complete
    print("Waiting for order to complete...")
    asyncio.run(wait_on_order(order_id))

    # Download order
    print("Downloading order")
    asyncio.run(download_order(order_id, outputs_path))
