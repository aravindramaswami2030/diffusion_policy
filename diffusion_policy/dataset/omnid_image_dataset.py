from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class OmnidImageDataset(BaseImageDataset):
    def __init__(self,
                 zarr_path,
                 shape_meta,
                 horizon,
                 pad_before,
                 pad_after,
                 n_obs_steps,
                 n_action_steps,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None
                 ):
        super().__init__()
        rgb_keys = list()
        low_dim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                low_dim_keys.append(key)
        obs_keys = rgb_keys + low_dim_keys

        # take n_obs steps worth of data
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in obs_keys:
                key_first_k[key] = n_obs_steps

        # Create replay buffer
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path=zarr_path,
            keys=obs_keys + ['action']
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)

        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs):
        normalizer = LinearNormalizer()

        # Action normalizer
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer['action'])

        # Low dim observations
        for key in self.low_dim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer[key])

        # Images
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        obs_dict = {}

        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(sample[key][0:self.n_obs_steps], -1, 1).astype(np.float32) / 255
            del sample[key]

        for key in self.low_dim_keys:
            obs_dict[key] = sample[key][0:self.n_obs_steps].astype(np.float32)
            del sample[key]

        data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(sample['action'].astype(np.float32))
        }
        return data


def test():

    zarr_path = '/home/aravind-linux/ws/omnid/src/omnid_ml/zarr_data/omnid_global.zarr'
    dataset = OmnidImageDataset(zarr_path, horizon=8)
    print("done")
    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

if __name__ == '__main__':
    test()
