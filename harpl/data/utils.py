import random
import numpy as np
import torch


class StatsRecorder:
    """Accumulates normalization statistics across mini-batches.
    ref: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html

    Args:
        red_dims (tuple): Dimensions to reduce over when computing statistics.
    """
    def __init__(self, red_dims=(0,1)):
        self.red_dims = red_dims # which mini-batch dimensions to average over
        self.nobservations = 0   # running number of observations

    def update(self, data):
        """ Updates the running statistics with a new mini-batch.

        Args:
            data (np.ndarray or torch.Tensor): New mini-batch of data of shape (nobservations, ndimensions).
        """
        # initialize stats and dimensions on first batch
        if self.nobservations == 0:
            self.mean = data.mean(dim=self.red_dims, keepdim=True)
            self.std  = data.std (dim=self.red_dims, keepdim=True)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            # find mean of new mini batch
            newmean = data.mean(dim=self.red_dims, keepdim=True)
            newstd  = data.std(dim=self.red_dims, keepdim=True)
            
            # update number of observations
            m = self.nobservations * 1.0
            n = data.shape[0]

            # update running statistics
            tmp = self.mean
            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean + 1e-5)**2
            self.std  = torch.sqrt(self.std)
            self.nobservations += n


def create_validation_sampler(dataset_size, validation_split, shuffle_dataset=True, distributed=False):
    """Creates a sampler for train and validation sets.

    Args:
        dataset_size (int): The size of the dataset.
        validation_split (float): The proportion of the dataset to use for validation.
        shuffle_dataset (bool): Whether to shuffle the dataset before splitting.

    Returns:
        torch.utils.data.sampler.SubsetRandomSampler: A sampler for the train set.
        torch.utils.data.sampler.SubsetRandomSampler: A sampler for the validation set.
    """
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    if distributed:
        raise NotImplementedError("HARPL keeps dataset loading single-process for MPS compatibility")
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


def seed_worker(worker_id):
    """Seed the worker for reproducibility.

    Args:
        worker_id (int): The worker ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
