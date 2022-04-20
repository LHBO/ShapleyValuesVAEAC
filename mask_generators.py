import numpy as np
import torch
from torchvision import transforms


# Mask generator for missing feature imputation

class MCARGenerator:
    """
    Returned mask is sampled from component-wise independent Bernoulli
    distribution with probability of component to be unobserved p.
    Such mask induces the type of missingness which is called
    in literature "missing completely at random" (MCAR).

    If some value in batch is missed, it automatically becomes unobserved.
    """
    def __init__(self, p):
        # set the probability for masking a value in the batch
        self.p = p

    def __call__(self, batch):
        # Find out if the batch contains any missing values, i.e. NANs
        nan_mask = torch.isnan(batch).float()

        # Create a matrix of zeros (1 - self.p = 80% chance) and
        # ones (self.p = 20% chance), of the same dimension as the batch.
        bernoulli_mask_numpy = np.random.choice(2, size=batch.shape,
                                                p=[1 - self.p, self.p])

        # Convert from numpy to torch tensor
        bernoulli_mask = torch.from_numpy(bernoulli_mask_numpy).float()

        # Create the final mask which combined nans and artificially masked values
        # Value of 1 means that the value is masked. Take max of the two masks
        mask = torch.max(bernoulli_mask, nan_mask)  # logical or
        return mask


class SpecifiedMaskGenerator:
    """
    Mask generator that takes in a matrix over possible masks and their probabilities.
    Then it samples masks from this matrix, with replacement, based on the given probabilities.

    If some value in batch is missed, it automatically becomes unobserved.
    """

    def __init__(self, mask_generator_only_these_coalitions, mask_generator_only_these_coalitions_probabilities):
        # Set the possible masks to sample from and the probabilities for each of them
        self.mask_generator_only_these_coalitions = mask_generator_only_these_coalitions
        self.mask_generator_only_these_coalitions_probabilities = \
            mask_generator_only_these_coalitions_probabilities / np.sum(mask_generator_only_these_coalitions_probabilities)
        self.n_coalitions, self.n_features = mask_generator_only_these_coalitions.shape

    def __call__(self, batch):
        # Find out if the batch contains any missing values, i.e., NANs.
        nan_mask = torch.isnan(batch).float()

        # Get the batch size and the number of explanatory variables.
        batch_size, num_variables = batch.size()

        # Create random number generator instance.
        rng = np.random.default_rng()

        # Sample the number of masked instances in each row.
        mask_rows = rng.choice(np.arange(self.n_coalitions),
                               size=batch_size,
                               replace=True,
                               p=self.mask_generator_only_these_coalitions_probabilities)

        # Extract the rows/masks from the matrix of possible masks
        # Convert from numpy to torch tensor
        mask = torch.from_numpy(self.mask_generator_only_these_coalitions[mask_rows, :]).float()

        # Create the final mask which combined nans and artificially masked values
        # Value of 1 means that the value is masked. Take max of the two masks
        mask = torch.max(mask, nan_mask)  # logical or

        return mask


class SpecifiedProbabilityGenerator:
    """
    A class that takes in the probabilities of having d masked observations.
    That is, if the data have M dimensions, then p_array is a M+1 long
    array, where p_array[d] is the probability of having d masked values.
    Recall that Python starts counting/indexing at 0.

    Note that MCARGenerator with p = 0.5 is the same as using
    SpecifiedProbabilityGenerator with
    p_array = np.array([math.comb(10, k) for k in range(10+1)]).

    I want to test out if increasing the probability of having a mask
    with many masked variables increase the performance of VAEAC, since
    VAEAC struggles a bit with these situations when only a few variables
    are known.
    p_array = probs = choose(10, 1:9) # The one currently used in VAEAC
    probs[c(1,2,3,4)] = probs[c(1,2,3,4)] * c(5,2,2,1.2)
    """
    def __init__(self, p_array):
        # p_array should be num_explanatory_variables - 1
        # Set the probability for masking a value in the batch.
        # Make sure that they sum to one.
        self.p_array = p_array / np.sum(p_array)

    def __call__(self, batch):
        # Find out if the batch contains any missing values, i.e., NANs.
        nan_mask = torch.isnan(batch).float()

        # Get the batch size and the number of explanatory variables.
        batch_size, num_variables = batch.size()

        # Create random number generator instance.
        rng = np.random.default_rng()

        # Sample the number of masked instances in each row.
        num_masked_each_row = rng.choice(np.arange(num_variables + 1),
                                         size=batch_size,
                                         replace=True,
                                         p=self.p_array)

        # Crate the mask matrix
        random_mask = torch.zeros(batch.size())
        for j in range(batch_size):
            random_mask[j, rng.choice(num_variables, size=num_masked_each_row[j], replace=False)] = 1

        # Create the final mask which combined nans and artificially masked values
        # Value of 1 means that the value is masked. Take max of the two masks
        mask = torch.max(random_mask, nan_mask)  # logical or
        return mask

