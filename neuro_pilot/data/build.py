import os
import random
from typing import Iterator
import numpy as np
import torch
from torch.utils.data import DataLoader, distributed

class InfiniteDataLoader(DataLoader):
    """DataLoader that reuses workers for infinite iteration.
    Reduces overhead of worker creation between epochs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        try:
            if not hasattr(self.iterator, "_workers"):
                return
            for w in self.iterator._workers:
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()
        except Exception:
            pass

    def reset(self):
        self.iterator = self._get_iterator()

class _RepeatSampler:
    """Sampler that repeats indefinitely."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        while True:
            yield from iter(self.sampler)

def seed_worker(worker_id):
    """Reproducibility seed for workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1, sampler=None, drop_last=False, pin_memory=True):
    """Professional dataloader factory supporting high-performance training.

    Args:
        dataset: Dataset or Subset to load from.
        batch: Batch size.
        workers: Number of workers.
        shuffle: Whether to shuffle.
        rank: Rank for distributed training.
        sampler: Optional custom sampler (overrides shuffle/distributed).
        drop_last: Whether to drop last batch.
        pin_memory: Whether to pin memory.
    """
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()
    nw = min(os.cpu_count() // max(nd, 1), workers)

    # Handle Subsets (from random_split) to find collate_fn
    collate_fn = getattr(dataset, "collate_fn", None)
    if collate_fn is None and hasattr(dataset, "dataset"):
         collate_fn = getattr(dataset.dataset, "collate_fn", None)

    if sampler is None and rank != -1:
        sampler = distributed.DistributedSampler(dataset, shuffle=shuffle)

    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=nd > 0 and pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        drop_last=drop_last and len(dataset) % batch != 0
    )
