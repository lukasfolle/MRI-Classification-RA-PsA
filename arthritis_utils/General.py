import os
import pickle
from monai import Compose
from monai.data import PersistentDataset, DataLoader


def path_exists_check(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file or path at {file_path}.")


def print_pickle(file_path):
    print(read_pickle(file_path))


def read_pickle(file_path):
    path_exists_check(file_path)
    with open(file_path, "rb") as file:
        return pickle.load(file)

def build_processing_pipeline(transformation_settings):
    return Compose([
        build_transformation(transform) for transform in transformation_settings
    ])


def build_transformation(transformation: dict):
    key = transformation.keys()
    if len(list(key)) > 1:
        raise ValueError("Only one key is valid for creation of a Transformation.")
    key = list(key)[0]
    return eval(key)(**transformation[key])


def get_persistent_dataset(db, transforms, cache: bool = True, cache_folder: str = "MONAI_no_contour"):
    db = db.get_db_monai_format()
    if cache:
        if not os.path.exists(cache_folder):
            raise FileNotFoundError(f"Could not find cache folder {cache_folder}.")
        print(f"INFO: Dataset will be cached in {cache_folder}")
    else:
        cache_folder = None
    persistent_ds = PersistentDataset(db, transform=transforms, cache_dir=cache_folder)
    return persistent_ds


class ArthritisDataLoader(DataLoader):
    pass

