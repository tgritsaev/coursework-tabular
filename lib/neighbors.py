from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

import delu

from . import util


def compute_knn(
    embeddings: dict[str, Tensor], n_neighbors: int, batch_size: int = 1024
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:  # (neighbors, distances)
    neighbors = {}
    distances = {}
    parts = ['train', 'val', 'test']

    while True:
        try:
            for part in parts:
                distances[part] = []
                neighbors[part] = []
                for i_batch in tqdm(
                    delu.data.make_index_dataloader(
                        len(embeddings[part]), batch_size=batch_size
                    ),
                    desc=f'part="{part}"',
                ):
                    batch_distances = torch.cdist(
                        embeddings[part][i_batch][None], embeddings['train'][None]
                    ).squeeze(0)
                    if part == 'train':
                        batch_distances[torch.arange(len(i_batch)), i_batch] = torch.inf
                    topk = torch.topk(
                        batch_distances, n_neighbors, dim=1, largest=False
                    )
                    distances[part].append(topk.values.cpu())
                    neighbors[part].append(topk.indices.cpu())
        except RuntimeError as err:
            if util.is_oom_exception(err):
                batch_size //= 2
                util.get_logger().warning(f'compute_knn: batch_size={batch_size}')
            else:
                raise
        else:
            break

    device = next(iter(embeddings.values())).device
    for data in [neighbors, distances]:
        for key in list(data):
            data[key] = torch.cat(data[key]).to(device)
    return neighbors, distances


def save_knn(
    neighbors: dict[str, Tensor], distances: dict[str, Tensor], path: Path
) -> None:
    assert path.is_dir()
    for part in neighbors:
        np.save(path / f'distances_{part}.npy', distances[part].cpu().numpy())
        np.save(
            path / f'neighbors_{part}.npy',
            neighbors[part].cpu().numpy().astype('int32'),  # to save space
        )
