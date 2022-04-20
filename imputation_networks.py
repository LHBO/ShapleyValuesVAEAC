from torch import nn
from torch.optim import Adam
import numpy as np

from mask_generators import MCARGenerator, SpecifiedProbabilityGenerator, SpecifiedMaskGenerator
from nn_utils import ResBlock, MemoryLayer, SkipConnection
from prob_utils import CategoricalToOneHotLayer, GaussianCategoricalLoss, \
    GaussianCategoricalSampler, SetGaussianSigmasToOne
from sys import stderr


def get_imputation_networks(one_hot_max_sizes,
                            p=0.2,
                            sample_most_probable=True,
                            width=256,
                            depth=10,
                            latent_dim=64,
                            lr=3e-4,
                            batch_size=64,
                            mask_generator_only_these_coalitions=None,
                            mask_generator_only_these_coalitions_probabilities=None
                            ):

    """
    This function builds neural networks for imputation given
    the list of one-hot max sizes of the dataset features.
    It returns a dictionary with those neural networks together with
    reconstruction log probability function, optimizer constructor,
    sampler from the generator output, mask generator, batch size,
    and scale factor for the stability of the variational lower bound
    optimization.   probabilities
    """

    # Proposal network
    proposal_layers = [
        # Here we do include add_nans_map_for_columns
        # as this network will receive the complete unaltered data,
        # hence, it can compute the nan.mask. I.e., if the data is missing
        CategoricalToOneHotLayer(one_hot_max_sizes +
                                 [0] * len(one_hot_max_sizes),
                                 list(range(len(one_hot_max_sizes)))),
        nn.Linear(sum(max(1, x) for x in one_hot_max_sizes) +
                  len(one_hot_max_sizes) * 2,
                  width),
        nn.LeakyReLU(),
    ]
    for i in range(depth):
        proposal_layers.append(
            SkipConnection(
                nn.Linear(width, width),
                nn.LeakyReLU(),
            )
        )
    proposal_layers.append(
        nn.Linear(width, latent_dim * 2)
    )
    proposal_network = nn.Sequential(*proposal_layers)

    # Prior network
    prior_layers = [
        # So here we do not include add_nans_map_for_columns
        # as this network will NOT receive the complete unaltered data,
        # both rather the observed values. So it does not
        # know if a 0 is missing (was a nan) or if it is just
        # masked.
        CategoricalToOneHotLayer(one_hot_max_sizes +
                                 [0] * len(one_hot_max_sizes)),
        MemoryLayer('#input'),
        nn.Linear(sum(max(1, x) for x in one_hot_max_sizes) +
                  len(one_hot_max_sizes),
                  width),
        nn.LeakyReLU(),
    ]
    for i in range(depth):
        prior_layers.append(
            SkipConnection(
                # skip-connection from prior network to generative network
                MemoryLayer('#%d' % i),
                nn.Linear(width, width),
                nn.LeakyReLU(),
            )
        )
    prior_layers.extend([
        MemoryLayer('#%d' % depth),
        nn.Linear(width, latent_dim * 2),
    ])
    prior_network = nn.Sequential(*prior_layers)

    # Generative network
    generative_layers = [
        nn.Linear(latent_dim, width),
        nn.LeakyReLU(),
    ]
    for i in range(depth + 1):
        generative_layers.append(
            SkipConnection(
                # skip-connection from prior network to generative network
                # Does a concatenation. So it concatenate the input of this
                # generative layer with a input to memory layer '#%d'.
                # Note that we add the memory layers in the opposite direction
                # from how they were created. So, we get a classical
                # U-net with latent space at the bottom and a connection
                # between the layers on the same height of the U-shape.
                MemoryLayer('#%d' % (depth - i), True),

                # We know need a width*2 input, as we have done the concatenation.
                nn.Linear(width * 2, width),
                nn.LeakyReLU(),
            )
        )
    generative_layers.extend([
        # Concatenate the input to the first layer of the prior network
        # to the last layer of the generative network.
        MemoryLayer('#input', True),
        nn.Linear(width + sum(max(1, x) for x in one_hot_max_sizes) +
                  len(one_hot_max_sizes),
                  sum(max(2, x) for x in one_hot_max_sizes)),
        # SetGaussianSigmasToOne(one_hot_max_sizes),
    ])
    generative_network = nn.Sequential(*generative_layers)

    # Small check if we are only sampling from a subset of possible masks
    if mask_generator_only_these_coalitions is not None:
        # Here we are in the new version where only a subset
        # of all possible coalitions is accessible.
        if not isinstance(mask_generator_only_these_coalitions, np.ndarray):
            raise ValueError("In 'get_imputation_networks()',"
                             " mask_generator_only_these_coalitions was not a numpy matrix.")

        # Have in the main code checked that the masks are of the correct dimensions
        mask_generator = SpecifiedMaskGenerator(
            mask_generator_only_these_coalitions=mask_generator_only_these_coalitions,
            mask_generator_only_these_coalitions_probabilities=mask_generator_only_these_coalitions_probabilities
        )

    else:
        # Here we are in the old version

        # Check if we are given a list of masking probabilities.
        # If yes, then use 'SpecifiedProbabilityGenerator', otherwise we
        # use 'MCARGenerator'.
        if isinstance(p, (list, np.ndarray)):
            if len(p) != sum(max(1, x) for x in one_hot_max_sizes) + 1:
                raise ValueError("In 'get_imputation_networks()', p was a list of wrong dimension.")
            else:
                mask_generator = SpecifiedProbabilityGenerator(p)
        else:
            mask_generator = MCARGenerator(p)

    return {
        'batch_size': batch_size,

        'reconstruction_log_prob': GaussianCategoricalLoss(one_hot_max_sizes),

        'sampler': GaussianCategoricalSampler(one_hot_max_sizes,
                                              sample_most_probable=sample_most_probable),

        'sampler_most_prob_true': GaussianCategoricalSampler(one_hot_max_sizes, sample_most_probable=True),

        'sampler_most_prob_false': GaussianCategoricalSampler(one_hot_max_sizes, sample_most_probable=False),

        # Not sure why we do this.
        # It should not effect the end result.
        # As we want to maximize the VLB, so it does not
        # matter if we maximize VLB or VLBxConstant.
        'vlb_scale_factor': 1 / len(one_hot_max_sizes),

        'optimizer': lambda parameters: Adam(parameters, lr=lr),

        'mask_generator': mask_generator,

        'proposal_network': proposal_network,

        'prior_network': prior_network,

        'generative_network': generative_network,

        'depth': depth,

        'width': width,

        'lr': lr,
    }


def get_imputation_networks_without_skip_connections(
        one_hot_max_sizes,
        p=0.2,
        sample_most_probable=True,
        width=256,
        depth=10,
        latent_dim=64,
        lr=3e-4,
        batch_size=64,
        mask_generator_only_these_coalitions=None,
        mask_generator_only_these_coalitions_probabilities=None):
    """
    This function builds neural networks for imputation given
    the list of one-hot max sizes of the dataset features.
    It returns a dictionary with those neural networks together with
    reconstruction log probability function, optimizer constructor,
    sampler from the generator output, mask generator, batch size,
    and scale factor for the stability of the variational lower bound
    optimization.

    Same as get_imputation_networks(), but this time without skip connections
    between the masked encoder and decoder.
    """

    # Proposal network
    proposal_layers = [
        # Here we do include add_nans_map_for_columns
        # as this network will receive the complete unaltered data,
        # hence, it can compute the nan.mask. I.e., if the data is missing
        CategoricalToOneHotLayer(one_hot_max_sizes +
                                 [0] * len(one_hot_max_sizes),
                                 list(range(len(one_hot_max_sizes)))),
        nn.Linear(sum(max(1, x) for x in one_hot_max_sizes) +
                  len(one_hot_max_sizes) * 2,
                  width),
        nn.ReLU(),
    ]
    for i in range(depth):
        # Removed Skip-connections
        proposal_layers.extend((nn.Linear(width, width), nn.ReLU()))
    proposal_layers.append(nn.Linear(width, latent_dim * 2))
    proposal_network = nn.Sequential(*proposal_layers)

    # Prior network
    prior_layers = [
        # So here we do not include add_nans_map_for_columns
        # as this network will NOT receive the complete unaltered data,
        # both rather the observed values. So it does not
        # know if a 0 is missing (was a nan) or if it is just
        # masked.
        CategoricalToOneHotLayer(one_hot_max_sizes +
                                 [0] * len(one_hot_max_sizes)),
        nn.Linear(sum(max(1, x) for x in one_hot_max_sizes) +
                  len(one_hot_max_sizes),
                  width),
        nn.ReLU(),
    ]
    for i in range(depth):
        prior_layers.extend((nn.Linear(width, width), nn.ReLU()))
    prior_layers.extend([nn.Linear(width, latent_dim * 2)])
    prior_network = nn.Sequential(*prior_layers)

    # Generative network
    generative_layers = [
        nn.Linear(latent_dim, width),
        nn.ReLU(),
    ]
    for i in range(depth):
        generative_layers.extend((
                # We now need a width input, as we do not concatenate with skip-connections
                nn.Linear(width, width),
                nn.ReLU()))
    generative_layers.extend([
        nn.Linear(width, sum(max(2, x) for x in one_hot_max_sizes)),
        # SetGaussianSigmasToOne(one_hot_max_sizes),
    ])
    generative_network = nn.Sequential(*generative_layers)

    # Small check if we are only sampling from a subset of possible masks
    if mask_generator_only_these_coalitions is not None:
        # Here we are in the new version where only a subset
        # of all possible coalitions is accessible.
        if not isinstance(mask_generator_only_these_coalitions, np.ndarray):
            raise ValueError("In 'get_imputation_networks()',"
                             " mask_generator_only_these_coalitions was not a numpy matrix.")

        # Have in the main code checked that the masks are of the correct dimensions
        mask_generator = SpecifiedMaskGenerator(
            mask_generator_only_these_coalitions=mask_generator_only_these_coalitions,
            mask_generator_only_these_coalitions_probabilities=mask_generator_only_these_coalitions_probabilities
        )

    else:
        # Here we are in the old version

        # Check if we are given a list of masking probabilities.
        # If yes, then use 'SpecifiedProbabilityGenerator', otherwise we
        # use 'MCARGenerator'.
        if isinstance(p, (list, np.ndarray)):
            if len(p) != sum(max(1, x) for x in one_hot_max_sizes) + 1:
                raise ValueError("In 'get_imputation_networks()', p was a list of wrong dimension.")
            else:
                mask_generator = SpecifiedProbabilityGenerator(p)
        else:
            mask_generator = MCARGenerator(p)

    return {
        'batch_size': batch_size,

        'reconstruction_log_prob': GaussianCategoricalLoss(one_hot_max_sizes),

        'sampler': GaussianCategoricalSampler(one_hot_max_sizes,
                                              sample_most_probable=sample_most_probable),

        'sampler_most_prob_true': GaussianCategoricalSampler(one_hot_max_sizes, sample_most_probable=True),

        'sampler_most_prob_false': GaussianCategoricalSampler(one_hot_max_sizes, sample_most_probable=False),

        'vlb_scale_factor': 1 / len(one_hot_max_sizes),

        'optimizer': lambda parameters: Adam(parameters, lr=lr),

        'mask_generator': mask_generator,

        'proposal_network': proposal_network,

        'prior_network': prior_network,

        'generative_network': generative_network,

        'depth': depth,

        'width': width,

        'lr': lr,
    }
