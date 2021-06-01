import geopandas
import pandas as pd
import numpy as np

from .base import BaseProcessor

from typing import List


class KenyaPVProcessor(BaseProcessor):
    dataset = "plant_village_kenya"

    def process(self) -> None:

        subfolders = [f"ref_african_crops_kenya_01_labels_0{i}" for i in [0, 1, 2]]

        dfs: List[geopandas.GeoDataFrame] = []
        for subfolder in subfolders:
            df = geopandas.read_file(
                self.raw_folder / "ref_african_crops_kenya_01_labels" / subfolder / "labels.geojson"
            )
            df = df.rename(
                columns={
                    "Latitude": "lat",
                    "Longitude": "lon",
                    "Planting Date": "planting_date",
                    "Estimated Harvest Date": "harvest_date",
                    "Crop1": "label",
                    "Survey Date": "collection_date",
                }
            )
            df["planting_date"] = pd.to_datetime(df["planting_date"]).dt.to_pydatetime()
            df["harvest_date"] = pd.to_datetime(df["harvest_date"]).dt.to_pydatetime()
            df["collection_date"] = pd.to_datetime(df["collection_date"]).dt.to_pydatetime()
            df["is_crop"] = np.where((df["label"] == "Fallowland"), 0, 1)
            df = df.to_crs("EPSG:4326")
            dfs.append(df)

        df = pd.concat(dfs)
        df = df.reset_index(drop=True)
        df["index"] = df.index
        df.to_file(self.output_folder / "data.geojson", driver="GeoJSON")
