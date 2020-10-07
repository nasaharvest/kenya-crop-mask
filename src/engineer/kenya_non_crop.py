from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import geopandas
from datetime import datetime
import numpy as np

from typing import Optional, Tuple


from src.processors import KenyaNonCropProcessor
from src.exporters import KenyaNonCropSentinelExporter
from .base import BaseEngineer, BaseDataInstance


@dataclass
class KenyaNonCropDataInstance(BaseDataInstance):
    is_crop: bool = False


class KenyaNonCropEngineer(BaseEngineer):

    sentinel_dataset = KenyaNonCropSentinelExporter.dataset
    dataset = KenyaNonCropProcessor.dataset

    @staticmethod
    def read_labels(data_folder: Path) -> pd.DataFrame:
        pv_kenya = data_folder / "processed" / KenyaNonCropProcessor.dataset / "data.geojson"
        assert pv_kenya.exists(), "Kenya Non Crop processor must be run to load labels"
        return geopandas.read_file(pv_kenya)

    def process_single_file(
        self,
        path_to_file: Path,
        nan_fill: float,
        max_nan_ratio: float,
        add_ndvi: bool,
        add_ndwi: bool,
        calculate_normalizing_dict: bool,
        start_date: datetime,
        days_per_timestep: int,
        is_test: bool,
        return_autoencoder_instances: bool,
        autoencoder_instances_per_label: int,
    ) -> Tuple[Optional[KenyaNonCropDataInstance], Optional[np.ndarray]]:
        r"""
        Return a tuple of np.ndarrays of shape [n_timesteps, n_features] for
        1) the anchor (labelled)
        """

        da = self.load_tif(path_to_file, days_per_timestep=days_per_timestep, start_date=start_date)

        # first, we find the label encompassed within the da

        min_lon, min_lat = float(da.x.min()), float(da.y.min())
        max_lon, max_lat = float(da.x.max()), float(da.y.max())
        overlap = self.labels[
            (
                (self.labels.lon <= max_lon)
                & (self.labels.lon >= min_lon)
                & (self.labels.lat <= max_lat)
                & (self.labels.lat >= min_lat)
            )
        ]
        if len(overlap) == 0:
            data_instance = None
            labelled_array = None
        else:
            label_lat = overlap.iloc[0].lat
            label_lon = overlap.iloc[0].lon

            closest_lon, lon_idx = self.find_nearest(da.x, label_lon)
            closest_lat, lat_idx = self.find_nearest(da.y, label_lat)

            labelled_np = da.sel(x=closest_lon).sel(y=closest_lat).values
            surrounding_np = self.get_surrounding_pixels(
                da, mid_lat_idx=lat_idx, mid_lon_idx=lon_idx
            )

            if add_ndvi:
                labelled_np = self.calculate_ndvi(labelled_np)
                surrounding_np = self.calculate_ndvi(surrounding_np)
            if add_ndwi:
                labelled_np = self.calculate_ndwi(labelled_np)
                surrounding_np = self.calculate_ndwi(surrounding_np)

            labelled_array = self.maxed_nan_to_num(
                labelled_np, nan=nan_fill, max_ratio=max_nan_ratio
            )
            surrounding_np = self.maxed_nan_to_num(surrounding_np, nan=nan_fill, max_ratio=None)

            if (not is_test) and calculate_normalizing_dict:
                self.update_normalizing_values(self.normalizing_dict_interim, labelled_array)
                self.update_normalizing_values(
                    self.surrounding_normalizing_dict_interim, surrounding_np
                )

            if labelled_array is not None:
                data_instance = KenyaNonCropDataInstance(
                    label_lat=label_lat,
                    label_lon=label_lon,
                    instance_lat=closest_lat,
                    instance_lon=closest_lon,
                    labelled_array=labelled_array,
                    surrounding_array=surrounding_np,
                )
            else:
                data_instance = None

        if return_autoencoder_instances:
            autoencoder_instances = self.process_autoencoder(
                da=da,
                normal_array=labelled_array,
                add_ndvi=add_ndvi,
                add_ndwi=add_ndwi,
                nan_fill=nan_fill,
                is_test=is_test,
                calculate_normalizing_dict=calculate_normalizing_dict,
                autoencoder_instances_per_label=autoencoder_instances_per_label,
            )
        else:
            autoencoder_instances = None

        return data_instance, autoencoder_instances