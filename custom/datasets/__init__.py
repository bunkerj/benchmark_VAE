import h5py
import os
import shutil
import random
import requests
import numpy as np
import torch
import torchvision
import dill as pickle
from tqdm import tqdm
from torch.utils.data import (
    Dataset, 
    DataLoader, 
    Sampler, 
    BatchSampler, 
    default_collate,
)

# `numpy_collate` and `NumpyLoader` are based on the JAX notebook https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html


def get_random_samples(u, x, ratio_rand_pts_enc):
    u = u.reshape(-1, 1)
    if ratio_rand_pts_enc == -1:
        return u, x, u, x

    n_total_pts = u.shape[0]
    n_rand_pts = int(ratio_rand_pts_enc * n_total_pts)
    indices = np.random.choice(n_total_pts, n_rand_pts, replace=False)
    indices_comp = np.setdiff1d(np.arange(n_total_pts), indices)

    u_enc = u[indices]
    x_enc = x[indices]

    u_dec = u[indices_comp]
    x_dec = x[indices_comp]

    return u_enc, x_enc, u_dec, x_dec


def get_raw_x(shape):
    x_mesh_list = np.meshgrid(
        np.linspace(0, 1, shape[-2] + 2)[1:-1], 
        np.linspace(0, 1, shape[-1] + 2)[1:-1], 
        indexing='ij',
    )
    xs = np.concatenate([np.expand_dims(v, -1) for v in x_mesh_list], axis=-1)
    xs = xs.reshape(-1, 2)
    return xs


class OnDiskDataset:
    @property
    def dataset_dir(self):
        """The path to the dataset directory on disk."""
        return os.path.join(os.getcwd(), self.data_base, "data", self.dataset_name)

    @property
    def dataset_path(self):
        """The path to the dataset on disk."""
        return os.path.join(self.dataset_dir, self.dataset_filename)

    @property
    def _data_exists(self):
        """Checks whether the dataset has already been downloaded to disk."""
        try:
            return os.path.isdir(self.dataset_dir)
        except AttributeError:
            raise NotImplementedError(
                "Dataset must have attributes data_base, dataset_name and dataset_filename"
            )


class GenerableDataset(Dataset, OnDiskDataset):
    """A dataset which can be generated locally and cached for future runs.

    `GenerableDataset` checks to see if the dataset has already been generated;
    if not present on disk, the dataset is generated using the method `generate`
    which is passed a handle to an HDF5 file.
    Otherwise, the dataset is loaded from the HDF5 file and stored in the attribute `data`.

    Derived classes must implement `generate`, `__len__` and `__getitem__`.
    """

    def __init__(self, train=True, data_base="", force_generate=False, *args, **kwargs):
        self.train = train
        self.data_base = data_base
        self.generated = False
        if not self._data_exists or force_generate:
            if os.path.exists(self.dataset_dir):
                print(f"Purging old dataset")
                shutil.rmtree(self.dataset_dir)

            print(f"Generating dataset {self.dataset_name}")
            os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
            with h5py.File(self.dataset_path, "w") as f:
                self.generate(f)
            print("Generation complete.")
            self.generated = True

        # TODO: hacky --- holds on to handle
        self.data = h5py.File(self.dataset_path, "r")

        super().__init__(*args, **kwargs)

    def generate(self, hdf5_file):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


class DownloadableDataset(Dataset, OnDiskDataset):
    """A dataset which manages downloading, preprocessing and loading from a remote source.

    `DownloadableDataset` checks to see if the dataset has already been downloaded;
    if not present on disk, the dataset is downloaded from the remote repository, as
    specified through the attribute `data_url`.
    The dataset is then preprocessed using the `_preprocess_data` method (which by default
    does nothing) and then in any case is loaded using `_load_data`.

    The preprocessing step is run by default after download, and can be used for tasks such as
    unzipping a compressed file. It can be forced using the `force_preprocess` option, which will
    run the preprocessing step even if the dataset is already present on disk.

    Derived classes must implement `_load_data`, `__len__` and `__getitem__`.
    They can also optionally implement `_preprocess_data`.

    Parameters
    ----------
    download : bool
        Toggles whether to allow downloading the dataset from the remote source. (default = `True`)
        Even if `True`, will only download if not already present on disk.

    force_preprocess : bool
        If `True`, runs the preprocessing step even if the dataset has already been downloaded. (default = `False`)

    force_download : bool
        If `True`, redownloads and preprocesses the data even if already downloaded. (default = `False`)

    train : bool
        Toggles whether to use train or test split. (default = `True`)

    data_base : str
        Modifies the search path for the data directory. (default = `''`)
        The default search path is `current_working_directory/data`, and
        `data_base` alters the base path relative to `current_working_directory`.
    """

    def __init__(
        self,
        download=True,
        force_preprocess=False,
        force_download=False,
        train=True,
        data_base="",
        *args,
        **kwargs,
    ):
        self.train = train
        self.data_base = data_base

        if not self._data_exists or force_download:
            if not download:
                raise ValueError(
                    "Dataset not found and download=False. Try setting download=True."
                )
            if os.path.exists(self.dataset_dir):
                print(f"Purging old dataset")
                shutil.rmtree(self.dataset_dir)

            print(f"Downloading dataset {self.dataset_name}.")
            self._download_data()
            print(f"Preprocessing dataset {self.dataset_name}")
            self._preprocess_data()
        elif force_preprocess:
            self._preprocess_data()

        self._load_data(train)
        super().__init__(*args, **kwargs)

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def _load_data(self, train):
        """Loads the downloaded and preprocessed dataset from disk.

        Parameters
        ----------

        train : bool
            Determines whether the train or test split should be loaded.
        """
        raise NotImplementedError()

    def _preprocess_data(self):
        """Preprocesses the downloaded dataset.

        For example, this can be used for extracting zipped files or other data
        preprocessing after download.

        The default action is to do no preprocessing.
        """
        pass

    def _download_data(self):
        """Downloads the dataset from the URL specified by attribute `data_url`."""
        try:
            os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
            r = requests.get(
                self.data_url, stream=True, headers={"User-agent": "Mozilla/5.0"}
            )
            if r.status_code != 200:
                raise ValueError(
                    f"Failed to download from specified data_url: error code {r.status_code}"
                )
            total = int(r.headers.get("content-length", 0))

            with open(self.download_path, "wb") as f, tqdm(
                total=total, unit="iB", unit_scale=True, unit_divisor=1024
            ) as bar:
                for data in r.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)

        except AttributeError:
            raise NotImplementedError(
                "DownloadableDataset must have attribute data_url"
            )

    @property
    def download_path(self):
        """The path to the downloaded file on disk."""
        return os.path.join(
            os.getcwd(),
            self.data_base,
            "data",
            self.dataset_name,
            self.download_filename,
        )


class Rescalable2DMixin:
    def __init__(self, resolution, *args, **kwargs):
        self.transform = torchvision.transforms.Resize(size=resolution)
        # Generate co-ordinate grid
        x1 = np.linspace(0, 1, resolution + 2)[1:-1]
        xs = np.meshgrid(*([x1] * 2), indexing="ij")
        xs = [np.expand_dims(v, -1) for v in xs]
        x = np.concatenate(xs, axis=-1)
        self.x = np.reshape(x, (-1, x.shape[-1]))
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = self._get_unscaled(idx)
        return np.reshape(
            self.transform(torch.tensor(np.expand_dims(item, 0))).numpy(),
            (-1, 1),
        )


class MultiBatchSampler(Sampler[int]):
    def __init__(self, datasets, batch_size, drop_last=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batch_indices_list = self._get_batch_indices_list(datasets, batch_size, drop_last)

    def __len__(self):
        return len(self.batch_indices_list)

    def __iter__(self):
        self.batch_indices_list = self._get_batch_indices_list(self.datasets, self.batch_size, self.drop_last)
        for batch_indices in self.batch_indices_list:
            yield batch_indices

    def _get_batch_indices_list(self, datasets, batch_size, drop_last):
        start_idx = 0
        batch_indices_list = []
        for dataset in datasets:
            end_idx = start_idx + len(dataset)
            indices = list(range(start_idx, end_idx))
            random.shuffle(indices)
            batch_indices = list(BatchSampler(indices, batch_size=batch_size, drop_last=drop_last))
            batch_indices_list += batch_indices
            start_idx += len(dataset)
        random.shuffle(batch_indices_list)
        return batch_indices_list


def pickle_save(obj, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as file:
        pickle.dump(obj, file)


def pickle_load(save_path):
    with open(save_path, 'rb') as file:
        return pickle.load(file)
