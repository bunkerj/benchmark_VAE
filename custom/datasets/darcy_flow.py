import os
import zipfile
import scipy
import torch
from custom.datasets import (
    DownloadableDataset, 
    get_raw_x, 
    pickle_save, 
    pickle_load,
)


class DarcyFlow(DownloadableDataset):
    '''
    This dataset is nearly identical to `cno.DarcyFlow`. See the documentation
    for `cno.DarcyFlow` for a more detailed discussion of the differences.
    Run once with save_fast=True to save the fast data files.
    Subsequent runs can be done with load_fast=True to load the fast data files.
    '''
    data_url = 'https://drive.google.com/u/0/uc?id=1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf&export=download&confirm=t&uuid=9d0c35a0-3979-4852-b8fd-c1d4afec423c&at=AB6BwCA0wHtyid20GZfaIBVJ4aQv:1702379684316'
    download_filename = 'Darcy_421.zip'
    dataset_filename = ''
    dataset_name = 'fno_darcy'
    
    def __init__(self, downscale=1, ratio_rand_pts_enc=-1,
                 save_fast=False, load_fast=False, *args, **kwargs):

        self.ratio_rand_pts_enc=ratio_rand_pts_enc
        self.downscale = downscale
        self.save_fast = save_fast
        self.load_fast = load_fast

        super().__init__(*args, **kwargs)

    def _preprocess_data(self):
        with zipfile.ZipFile(self.download_path, 'r') as f:
            f.extractall(self.dataset_dir)

    def _get_slow_data_filename(self, train):
        if train:
            return 'piececonst_r421_N1024_smooth1.mat'
        else:
            return 'piececonst_r421_N1024_smooth2.mat'

    def _get_fast_data_filename(self, train):
        slow_data_filename = self._get_slow_data_filename(train)
        fast_data_filename = slow_data_filename.replace('.mat', '_fast.pkl')
        return fast_data_filename

    def _load_data(self, train):
        if self.load_fast:
            self._load_data_fast(train)
        else:
            self._load_data_slow(train)

    def _load_data_slow(self, train):
        self.dataset_filename = self._get_slow_data_filename(train)

        mat = scipy.io.loadmat(self.dataset_path, variable_names=['coeff', 'sol'])
        u = mat['sol'].astype(float)
        u = u[:, ::self.downscale, ::self.downscale]
        x = get_raw_x(u.shape[1:])

        u = u.reshape(u.shape[0], -1, 1)
        u = (u - u.min()) / (u.max() - u.min())
        self.data = {
            'u': u,
            'x': x,
        }

        if self.save_fast:
            print('Saving fast data')

            save_filename = self._get_fast_data_filename(train)
            save_path = os.path.join(self.dataset_dir, save_filename)
            pickle_save(self.data, save_path)

            print('Done!')

    def _load_data_fast(self, train):
        self.dataset_filename = self._get_fast_data_filename(train)
        load_path = os.path.join(self.dataset_dir, self.dataset_filename)
        self.data = pickle_load(load_path)

    def __len__(self):
        return self.data['u'].shape[0]

    def __getitem__(self, idx):
        u = self.data['u'][idx]
        # x = self.data['x']
        return torch.tensor(u, dtype=torch.float) #, x
