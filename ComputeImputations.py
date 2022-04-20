# This is going to be a function that takes in all possible coalitions
# for a single person at the time.
# So the input file has to be [2^p, p], where p is the dimension.
# We will likely stick to p = 3 or p = 10.
# Might be that we rather input 2^p - 2 rows as the extreme cases
# of all variables known or unknown are not interesting.
# However, this will not effect the code.


# Just include all the libraries. REMOVE THE UNUSED AT THE END
# from argparse import ArgumentParser
from copy import deepcopy
# from importlib import import_module
from math import ceil
from os.path import join
from os import replace
from sys import stderr

# %%
import numpy as np
import torch
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.functional import softplus, softmax

# %%
from tqdm import tqdm
import warnings

# %%

from imputation_networks import get_imputation_networks, get_imputation_networks_without_skip_connections
from datasets2 import compute_normalization
from train_utils import extend_batch, get_validation_iwae
from VAEAC import VAEAC


# %%
def train_VAEAC_model(data_train,
                      distribution,
                      param_now,
                      path_to_save_model,
                      one_hot_max_sizes,
                      use_cuda=False,
                      num_different_vaeac_initiate=5,
                      epochs_initiation_phase=2,
                      epochs=100,
                      masking_ratio=0.5,  # Can also be a list of sum(one_hot_max_sizes)
                      validation_ratio=0.25,
                      validation_iwae_num_samples=40,
                      validations_per_epoch=1,
                      width=32,
                      depth=3,
                      latent_dim=8,
                      lr=0.001,
                      batch_size=64,
                      running_avg_num_values=5,
                      use_skip_connections=True,
                      mask_generator_only_these_coalitions=None,
                      mask_generator_only_these_coalitions_probabilities=None,
                      verbose=False,
                      verbose_init=False,
                      verbose_summary=False
                      ):
    """
    Function that fits a VAEAC-model to the given dataset,
    based on the neural network architectures determined by the parameters above.
    One_hot_max_sizes is one for continuous variables and the number of categories
    for a categorical variable.

    The function returns a list of elements, where the first three elements are
    path-locations to three different fitted VAEAC-models. The one with lowest
    validation IWAE, the one with lowest running average validation IWAE, and
    the one after the last epoch.
    The forth to sixth elements are arrays of training variational lower bound,
    validation IWAE and running average validation IWAE.

    IWAE = Importance Sampling Estimator


        use_cuda = False
    num_different_vaeac_initiate = 5
    epochs_initiation_phase = 2
    epochs = 100
    masking_ratio = 0.5  # Can also be a list of sum(one_hot_max_sizes)
    validation_ratio = 0.25
    validation_iwae_num_samples = 40
    validations_per_epoch = 1
    width = 32
    depth = 3
    latent_dim = 8
    lr = 0.001
    batch_size = 64
    running_avg_num_values = 5
    use_skip_connections = True
    verbose = False
    verbose_init = False
    verbose_summary = False

    one_hot_max_sizes = [1, 1, 1]
    data_train = np.matrix([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])

    """
    # print(one_hot_max_sizes, file=stderr, flush=True)
    # print(type(one_hot_max_sizes), file=stderr, flush=True)
    # print(data_train, file=stderr, flush=True)
    # print(type(data_train), file=stderr, flush=True)


    # %%

    # If we are asked to run on cuda, check if it is possible
    cuda_available = torch.cuda.is_available()

    # Give warning to user if asked to run on cuda, but cuda is not available.
    if cuda_available is False and use_cuda is True:
        warnings.warn("Cuda is not available. Fall back to CPU.", ResourceWarning)

    # Define boolean whether we use cuda or not, then use CPU.
    use_cuda = cuda_available and use_cuda

    # Check that masking_ratio is of valid length
    if isinstance(masking_ratio, (list, np.ndarray)):
        print('Using SpecifiedProbabilityGenerator...', file=stderr, flush=True)
        masking_ratio = np.array(masking_ratio)

        # I THINK THIS IS INCORRECT.
        # THE MASKS SHOULD BE OF LENGTH len(one_hot_max_sizes),
        # AS WE MASK OUT WHOLE FEATURES AND NOT JUST SOME OF THE
        # ONE HOT ENCODINGS.
        # THIS MUST BE FIXED!
        correct_masking_ration_len = sum(max(1, x) for x in one_hot_max_sizes) + 1
        if len(masking_ratio) != correct_masking_ration_len:
            raise ValueError(f"Masking_ratio is a list of wrong length. "
                             f"It is of length {len(masking_ratio)}, but should be "
                             f"{correct_masking_ration_len}.")

    if mask_generator_only_these_coalitions is not None:
        print('Using SpecifiedMaskGenerator...', file=stderr, flush=True)
        mask_generator_only_these_coalitions = np.array(mask_generator_only_these_coalitions)
        mask_generator_only_these_coalitions_probabilities = np.array(mask_generator_only_these_coalitions_probabilities)

        n_coalitions, n_features = mask_generator_only_these_coalitions.shape
        n_probabilities = len(mask_generator_only_these_coalitions_probabilities)
        if n_probabilities != n_coalitions or n_features != len(one_hot_max_sizes):
            raise ValueError(f"Input to SpecifiedMaskGenerator is incorrect.")


    # %%
    # Get the number of observation and the dimension.
    data_train = np.array(data_train)
    n, p = data_train.shape

    # Convert the data from numpy to a torch
    raw_data = torch.from_numpy(data_train).float()

    # %%
    # Compute the mean and std for each continuous feature in the data
    # The categorical features will have mean zero and sd 1.
    # So we do not change the categorical features in the next two line of codes.
    # Ie. we subtract zero and divide by 1.
    norm_mean, norm_std = compute_normalization(raw_data, one_hot_max_sizes)

    # %%
    # Make sure that the standard deviation is not too high, in that case clip it.
    norm_std = torch.max(norm_std, torch.tensor(1e-9))

    # normalize the data to have mean = 0 and std = 1.
    data = (raw_data - norm_mean[None]) / norm_std[None]

    # %%
    # Non-zero number of workers cause nasty warnings because of some bug in
    # multiprocess library. It might be fixed now, but anyway there is no need
    # to have a lot of workers for dataloader over in-memory tabular data.
    num_workers = 0

    # %%
    # Splitting the input data into training and validation sets
    # Find the number of instances in the validation set
    val_size = ceil(len(data) * validation_ratio)
    # randomly sample indices for the validation set
    val_indices = np.random.choice(len(data), val_size, False)
    # Make it into a set such that we can find the training indices
    val_indices_set = set(val_indices)
    # Get the indices that are not in the validation set.
    train_indices = [i for i in range(len(data)) if i not in val_indices_set]
    # Create the training and validation set based on the corresponding indices.
    train_data = data[train_indices]
    val_data = data[val_indices]

    # %%
    if len(train_indices) <= batch_size:
        batch_size = len(train_indices) - 1
        print('Override batch size due to insufficient number of training observations.',
              file=stderr, flush=True)

    # %%
    # Initialize the dataloaders
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=False)

    # number of batches after which it is time to do validation
    # should not be ceil, as ceil(67 / 3) * 3 = 69, so we would only get
    # two validations per epoch. Does not matter as val_per_epoch == 1.
    validation_batches = ceil(len(dataloader) / validations_per_epoch)

    # %%
    # Will now initialize several VAEAC. Train them for some epochs
    # and keep the one with the best training variational lower bound.
    # This is done iteratively, and keep the new vaeac if it has higher
    # vlb than 'best_vlb'.
    best_vlb = -10e20

    # %%
    for initialization in range(num_different_vaeac_initiate):
        if verbose_init:
            print('Initialising VAEAC number %d...' % (initialization + 1),
                  file=stderr, flush=True)

        if use_skip_connections:
            #print('Using skip connections... ', file=stderr, flush=True)
            networks = get_imputation_networks(
                one_hot_max_sizes=one_hot_max_sizes,
                sample_most_probable=False,
                p=masking_ratio,
                width=width,
                depth=depth,
                latent_dim=latent_dim,
                lr=lr,
                batch_size=batch_size,
                mask_generator_only_these_coalitions=mask_generator_only_these_coalitions,
                mask_generator_only_these_coalitions_probabilities=mask_generator_only_these_coalitions_probabilities)

        else:
            #print('NOT using skip connections... ', file=stderr, flush=True)
            networks = get_imputation_networks_without_skip_connections(
                one_hot_max_sizes=one_hot_max_sizes,
                sample_most_probable=False,
                p=masking_ratio,
                width=width,
                depth=depth,
                latent_dim=latent_dim,
                lr=lr,
                batch_size=batch_size,
                mask_generator_only_these_coalitions=mask_generator_only_these_coalitions,
                mask_generator_only_these_coalitions_probabilities=mask_generator_only_these_coalitions_probabilities
            )
            #print(networks, file=stderr, flush=True)


        # Build VAEAC on top of returned network
        model = VAEAC(
            networks['reconstruction_log_prob'],
            networks['proposal_network'],
            networks['prior_network'],
            networks['generative_network']
        )
        #print([depth, width], file=stderr, flush=True)
        #print(model, file=stderr, flush=True)

        # Load parameters from the network/model
        optimizer = networks['optimizer'](model.parameters())
        batch_size = networks['batch_size']
        mask_generator = networks['mask_generator']
        vlb_scale_factor = networks.get('vlb_scale_factor', 1)
        # The 1 in networks.get('vlb_scale_factor', 1) means that if
        # 'vlb_scale_factor' is not defined, return "1". This should not
        # be necessary for us.

        # A list of validation IWAE estimates
        validation_iwae = []
        validation_iwae_running_avg = []

        # A list of running variational lower bounds on the train set
        train_vlb = []
        # the length of two lists above is the same because the new
        # values are inserted into them at the validation checkpoints only

        # Start the training loop
        for epoch in range(epochs_initiation_phase):

            # %%
            # Set iterator to be the dataloader which loads the training data.
            iterator = dataloader

            # Set average variational lower bound to 0 for this epoch
            avg_vlb = 0

            # Print information to the user
            if verbose_init:
                print('Epoch %d...' % (epoch + 1), file=stderr, flush=True)
                # Wrap the iterator by tqdm which makes progress bars in the console
                iterator = tqdm(iterator)

            # %%
            # one epoch
            for i, batch in enumerate(iterator):
                # the time to do a checkpoint is at start and end of the training
                # and after processing validation_batches batches
                if any([i == 0 and epoch == 0,
                        i % validation_batches == validation_batches - 1,
                        i + 1 == len(dataloader)]):

                    # Compute the validation iwae
                    val_iwae = get_validation_iwae(val_dataloader,
                                                   mask_generator,
                                                   batch_size,
                                                   model,
                                                   validation_iwae_num_samples,
                                                   verbose_init)

                    # Add the current validation_iwae and train_vlb to the lists.
                    validation_iwae.append(val_iwae)
                    train_vlb.append(avg_vlb)
                    validation_iwae_running_avg.append(np.array(validation_iwae)[-running_avg_num_values:].mean())

                    if verbose_init:
                        print(file=stderr)
                        print(file=stderr)

                # If batch size is less than batch_size, extend it with objects from the beginning of the dataset
                batch = extend_batch(batch, dataloader, batch_size)

                # Generate mask and do an optimizer step over the mask and the batch
                mask = mask_generator(batch)

                # Set previous gradients to zero
                optimizer.zero_grad()

                # Compute the variational lower bound for the batch given the mask
                vlb = model.batch_vlb(batch, mask).mean()

                # Backpropagation: minimize the negative vlb.
                (-vlb / vlb_scale_factor).backward()

                # Update the model parameters by using ADAM.
                optimizer.step()

                # Update running variational lower bound average
                avg_vlb += (float(vlb) - avg_vlb) / (i + 1)

                # Print some information to the console
                if verbose_init:
                    iterator.set_description('Train VLB: %g' % avg_vlb)

            # %%
        # Get the best vaeac models of the initiated versions
        if best_vlb <= avg_vlb or initialization == 0:
            best_vlb = avg_vlb
            best_iteration = initialization + 1
            # The current fitted model has the best fit
            best_networks = networks
            best_model = model
            best_validation_iwae = validation_iwae
            best_validation_iwae_running_avg = validation_iwae_running_avg
            best_train_vlb = train_vlb
            best_optimizer = optimizer
            best_batch_size = batch_size
            best_mask_generator = mask_generator
            best_vlb_scale_factor = vlb_scale_factor

        if verbose_init:
            print('Training vlb of %.3f.' % (train_vlb[-1]), file=stderr, flush=True)

    # %%
    if verbose_summary:
        print('Of the %d initiations of VAEAC, number %d was the best\n'
              'with training vlb of %.3f after %d epochs.' % (num_different_vaeac_initiate,
                                                               best_iteration,
                                                               best_train_vlb[-1],
                                                               epochs_initiation_phase), file=stderr, flush=True)
    # Set the names of the best versions
    networks = best_networks
    model = best_model
    validation_iwae = best_validation_iwae
    validation_iwae_running_avg = best_validation_iwae_running_avg
    train_vlb = best_train_vlb
    optimizer = best_optimizer
    batch_size = best_batch_size
    mask_generator = best_mask_generator
    vlb_scale_factor = best_vlb_scale_factor

    # %%
    # Send the model to the GPU, if we have access to it.
    if use_cuda:
        model = model.cuda()

    # best model state according to the validation IWAE
    best_state = None

    # If we should print output in the main fitting loop
    # is updated if 'verbose' is true.
    verbose_main = False

    # %%
    # Start the training loop
    for epoch in range(epochs_initiation_phase, epochs):
        if verbose:
            if (epoch + 1) % (epochs / 10) == 0:
                verbose_main = True
            else:
                verbose_main = False

        # Set iterator to be the dataloader which loads the training data.
        iterator = dataloader

        # Set average variational lower bound to 0 for this epoch
        avg_vlb = 0

        # Print information to the user
        if verbose_main:
            print('Depth: ' + str(depth) + '\tWidth: ' + str(width) + '\tLatent Dim: ' + str(latent_dim)
                  + '\tLearning Rate: ' + str(lr), file=stderr, flush=True)
            print('Epoch %d...' % (epoch + 1), file=stderr, flush=True)
            # Wrap the iterator by tqdm which makes progress bars in the console
            iterator = tqdm(iterator)

        # one epoch
        for i, batch in enumerate(iterator):
            # the time to do a checkpoint is at start and end of the training
            # and after processing validation_batches batches
            if any([
                i == 0 and epoch == 0,
                i % validation_batches == validation_batches - 1,
                i + 1 == len(dataloader)
            ]):

                # Compute the validation iwae, which I am not certain what is.
                # IWAE is an abbreviation for Importance Sampling Estimator
                # log p_{theta, psi}(x|y) \approx
                # log {1/S * sum_{i=1}^S [p_theta(x|z_i, y) * p_psi(z_i|y) / q_phi(z_i|x,y)]}
                # where z_i ~ q_phi(z|x,y)
                val_iwae = get_validation_iwae(val_dataloader,
                                               mask_generator,
                                               batch_size,
                                               model,
                                               validation_iwae_num_samples,
                                               verbose_main)

                # Add the current validation_iwae and train_vlb to the lists.
                validation_iwae.append(val_iwae)
                train_vlb.append(avg_vlb)
                val_iwae_running_mean = np.array(validation_iwae)[-running_avg_num_values:].mean()
                validation_iwae_running_avg.append(val_iwae_running_mean)

                # if current model validation IWAE is the best validation IWAE
                # over the history of training, the current state
                # is saved to best_state variable
                # Dont think one need to flip the values in validation_iwae with -1.
                if max(validation_iwae[::-1]) <= val_iwae:
                    best_state = deepcopy({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'validation_iwae': validation_iwae,
                        'validation_iwae_running_avg': validation_iwae_running_avg,
                        'running_avg_num_values': running_avg_num_values,
                        'train_vlb': train_vlb,
                        'norm_mean': norm_mean,
                        'norm_std': norm_std,
                        'distribution': distribution,
                        'n': n,
                        'p': p,
                        'param_now': param_now,
                        'one_hot_max_sizes': one_hot_max_sizes,
                        'epochs': epochs,
                        'masking_ratio': masking_ratio,
                        'validation_ratio': validation_ratio,
                        'validation_iwae_num_samples': validation_iwae_num_samples,
                        'validations_per_epoch': validations_per_epoch,
                        'num_different_vaeac_initiate': num_different_vaeac_initiate,
                        'epochs_initiation_phase': epochs_initiation_phase,
                        'width': width,
                        'depth': depth,
                        'latent_dim': latent_dim,
                        'lr': lr,
                        'batch_size': batch_size,
                        'use_skip_connections': use_skip_connections
                    })
                    aux_str = str.lower(distribution) + '_p_' + str(p) + '_param_' + str(param_now) + \
                              '_n_' + str(n) + '_depth_' + str(depth) + '_width_' + str(width) + \
                              '_latent_' + str(latent_dim) + '_lr_' + str(lr) + '_best.pt'
                    filename_best = join(path_to_save_model, aux_str)
                    torch.save(best_state, filename_best + '.bak')
                    replace(filename_best + '.bak', filename_best)

                if max(validation_iwae_running_avg) <= val_iwae_running_mean or not "best_state_running" in locals():
                    best_state_running = deepcopy({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'validation_iwae': validation_iwae,
                        'validation_iwae_running_avg': validation_iwae_running_avg,
                        'running_avg_num_values': running_avg_num_values,
                        'train_vlb': train_vlb,
                        'norm_mean': norm_mean,
                        'norm_std': norm_std,
                        'distribution': distribution,
                        'n': n,
                        'p': p,
                        'param_now': param_now,
                        'one_hot_max_sizes': one_hot_max_sizes,
                        'epochs': epochs,
                        'masking_ratio': masking_ratio,
                        'validation_ratio': validation_ratio,
                        'validation_iwae_num_samples': validation_iwae_num_samples,
                        'validations_per_epoch': validations_per_epoch,
                        'num_different_vaeac_initiate': num_different_vaeac_initiate,
                        'epochs_initiation_phase': epochs_initiation_phase,
                        'width': width,
                        'depth': depth,
                        'latent_dim': latent_dim,
                        'lr': lr,
                        'batch_size': batch_size,
                        'use_skip_connections': use_skip_connections
                    })
                    aux_str = str.lower(distribution) + '_p_' + str(p) + '_param_' + str(param_now) + \
                              '_n_' + str(n) + '_depth_' + str(depth) + '_width_' + str(width) + \
                              '_latent_' + str(latent_dim) + '_lr_' + str(lr) + '_best_running.pt'
                    filename_best_running = join(path_to_save_model, aux_str)
                    torch.save(best_state_running, filename_best_running + '.bak')
                    replace(filename_best_running + '.bak', filename_best_running)

                if verbose_main:
                    print(file=stderr)
                    # print(file=stderr)

            # If batch size is less than batch_size, extend it with objects
            # from the beginning of the dataset
            batch = extend_batch(batch, dataloader, batch_size)

            # Generate mask and do an optimizer step over the mask and the batch
            # 20% for ones (masked) and 80% for zeros(observed)
            # in addition to those that get masked since they are missing.
            mask = mask_generator(batch)

            # Now we are going to do the training by minimizing the variational lower bound
            # Sets gradients of all model parameters to zero before starting to
            # do backpropragation because PyTorch accumulates the gradients
            # on subsequent backward passes
            optimizer.zero_grad()

            # Send the batch and mask to Nvida GPU if we have. Would be faster.
            if use_cuda:
                batch = batch.cuda()
                mask = mask.cuda()

            # Compute the variational lower bound for the batch
            # given the mask (which masks both nans and artificially nans)
            # Take the mean of the variational lower bound for each of the
            # instances in the batch.
            vlb = model.batch_vlb(batch, mask).mean()

            # We want to maximize the vlb, but adam minimize, so to get
            # a loss function we need to take the negative version.
            # Accumulate (i.e. sum) all gradients of the loss function
            # back to the original tensors, i.e., backpropagation.
            # The loss is given by, in the cont case:
            # loss = -vlb / vlb_scale_factor = -vlb * len(one_hot_max_sizes)
            (-vlb / vlb_scale_factor).backward()

            # Use adam (Adaptive Moment Estimation) optimization routine to compute the update for the model parameters
            # based on the current gradient (stored in .grad attribute of a parameter)
            # That is, for the proposal (encoder), the generative (decoder) and prior network.
            optimizer.step()

            # update running variational lower bound average
            # a + (new - a)/(i+1) = {(i+1)a + new - a}/(i+1) = { a(i) + new}/(i+1) = a *i/(i+1) + new/(i+1)
            # recursive average formula/update
            avg_vlb += (float(vlb) - avg_vlb) / (i + 1)

            # Print some information to the console
            if verbose_main:
                # Change the description (text before progress bar),
                # so that the user see that we now are training.
                iterator.set_description('Train VLB: %g' % avg_vlb)

    # Also save the last fitted model
    last_state = deepcopy({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_iwae': validation_iwae,
        'validation_iwae_running_avg': validation_iwae_running_avg,
        'running_avg_num_values': running_avg_num_values,
        'train_vlb': train_vlb,
        'norm_mean': norm_mean,
        'norm_std': norm_std,
        'distribution': distribution,
        'n': n,
        'p': p,
        'param_now': param_now,
        'one_hot_max_sizes': one_hot_max_sizes,
        'epochs': epochs,
        'masking_ratio': masking_ratio,
        'validation_ratio': validation_ratio,
        'validation_iwae_num_samples': validation_iwae_num_samples,
        'validations_per_epoch': validations_per_epoch,
        'num_different_vaeac_initiate': num_different_vaeac_initiate,
        'epochs_initiation_phase': epochs_initiation_phase,
        'width': width,
        'depth': depth,
        'latent_dim': latent_dim,
        'lr': lr,
        'batch_size': batch_size,
        'use_skip_connections': use_skip_connections
    })

    aux_str = str.lower(distribution) + '_p_' + str(p) + '_param_' + str(param_now) + \
              '_n_' + str(n) + '_depth_' + str(depth) + '_width_' + str(width) + \
              '_latent_' + str(latent_dim) + '_lr_' + str(lr) + '_last.pt'
    filename_last = join(path_to_save_model, aux_str)
    torch.save(last_state, filename_last + '.bak')
    replace(filename_last + '.bak', filename_last)

    if verbose_summary:
        print("\nBest epoch:             %d. \tVLB = %.4f. \tIWAE = %.4f \tIWAE_running = %.4f."
              "\nBest running avg epoch: %d. \tVLB = %.4f. \tIWAE = %.4f \tIWAE_running = %.4f."
              "\nLast epoch:             %d. \tVLB = %.4f. \tIWAE = %.4f \tIWAE_running = %.4f."
              % (best_state['epoch'] + 1,
                 best_state['train_vlb'][-1],
                 best_state['validation_iwae'][-1],
                 best_state['validation_iwae_running_avg'][-1],
                 best_state_running['epoch'] + 1,
                 best_state_running['train_vlb'][-1],
                 best_state_running['validation_iwae'][-1],
                 best_state_running['validation_iwae_running_avg'][-1],
                 last_state['epoch'] + 1,
                 last_state['train_vlb'][-1],
                 last_state['validation_iwae'][-1],
                 last_state['validation_iwae_running_avg'][-1]
                 ), file=stderr, flush=True)

    return filename_best, filename_best_running, filename_last, \
           np.array(train_vlb), np.array(validation_iwae), \
           np.array(validation_iwae_running_avg)


def VAEAC_training_vlb_and_validation_iwae(path_VAEAC_model):
    """
    A function that reads the python dictionary/VAEAC
    from a given path and returns the development of the
    training variational lower bound and the validation
    Importance Sampling Estimator.
    I.e., training and validation loss.

    path_VAEAC_model: string containing the location of the saved VAEAC model
    """
    print(path_VAEAC_model)
    # Load the VAEAC model at the provided path.
    # This loads a dictionary that contains the following elements:
    # 'epoch', 'model_state_dict', 'optimizer_state_dict',
    # 'validation_iwae', 'train_vlb', 'norm_mean', and 'norm_std'.
    checkpoint = torch.load(path_VAEAC_model)

    # Only want to return epoch, training loss, and validation loss.
    include_keys = ['validation_iwae', 'train_vlb', 'epoch']

    # Creat a dictionary of only the desired elements
    sub_dict = {k: checkpoint[k] for k in include_keys if k in checkpoint}

    """
    # Here is how to plot in python
    plt.plot(np.arange(10, len(checkpoint["train_vlb"]))+1, checkpoint["train_vlb"][10:], label="Training")
    plt.plot(np.arange(10, len(checkpoint["validation_iwae"]))+1, checkpoint["validation_iwae"][10:], label="Validation")
    plt.title("Training VLB and Validation IWAE")
    plt.xlabel("Epoch")
    plt.ylabel("VLB and IWAE")
    plt.legend(loc="lower right")
    plt.show()
    """
    return sub_dict

def VAEAC_latent_space(instances_to_impute,
                       path_VAEAC_model,
                       one_hot_max_sizes,
                       use_skip_connections=True,
                       use_cuda=False,
                       verbose=False):
    """
    This function loads the given VAEAC model.
    Run the instances to impute through the proposal network (encoder) and
    the prior network (the one we use at test time) to obtain the distributions
    in latent space. I.e., the means and the standard deviations of independent
    normal distributions.

    We return these means and standard deviations for both the proposal and prior network.

    This function follows the same code structure as 'VAEAC_impute_values()'.

    instances_to_impute: a 2^p (maybe 2^p-2) x p matrix in R. Gets automatically
                         transformed to a numpy.ndarray when sent to python.
                         Should be 'nan' (or 'NaN' in R) where we want to impute
                         the missing values.
    path_VAEAC_model: string containing the location of the saved VAEAC model
    one_hot_max_sizes: array with the one-hot max sizes for categorical features and 0 or 1
                       for real-valued ones. We deal only with real-valued. The length of
                       the array must be equal to the number of columns/dimension of covariates.
    """

    # If we are asked to run on cuda, check if it is possible
    cuda_available = torch.cuda.is_available()

    # Give warning to user if asked to run on cuda, but cuda is not available.
    if cuda_available is False and use_cuda is True:
        warnings.warn("Cuda is not available. Fall back to CPU.", ResourceWarning)

    # Define boolean whether we use cuda or not, then use CPU.
    use_cuda = cuda_available and use_cuda

    # Load the VAEAC model at the provided path.
    # This loads a dictionary that contains the following elements:
    # 'epoch', 'model_state_dict', 'optimizer_state_dict',
    # 'validation_iwae', 'train_vlb', 'norm_mean', and 'norm_std'.
    checkpoint = torch.load(path_VAEAC_model)

    if use_skip_connections:
        # Extract some of the parameters based on the dimensions of the networks
        depth = int(list(checkpoint['model_state_dict'].keys())[-1].split(".")[1]) - 4
        width, latent_dim = checkpoint['model_state_dict']['generative_network.0.weight'].shape
        lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']

        # Design all necessary networks and learning parameters for the dataset
        # Might be possible to eliminate some stuff here to make it faster
        # as we are not going to continue training.
        networks = get_imputation_networks(one_hot_max_sizes=one_hot_max_sizes,
                                           width=width,
                                           depth=depth,
                                           latent_dim=latent_dim,
                                           lr=lr)
    else:
        # Extract some of the parameters based on the dimensions of the networks
        depth = int(list(checkpoint['model_state_dict'].keys())[-1].split(".")[1]) - 5
        width, latent_dim = checkpoint['model_state_dict']['generative_network.0.weight'].shape
        lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']

        # Design all necessary networks and learning parameters for the dataset
        # Might be possible to eliminate some stuff here to make it faster
        # as we are not going to continue training.
        networks = get_imputation_networks_without_skip_connections(
            one_hot_max_sizes=one_hot_max_sizes,
            width=width,
            depth=depth,
            latent_dim=latent_dim,
            lr=lr)

    # Crate the VAEAC model by providing the necessary networks
    model = VAEAC(
        networks['reconstruction_log_prob'],
        networks['proposal_network'],
        networks['prior_network'],
        networks['generative_network']
    )

    # Send the model to the GPU, if we are supposed to.
    if use_cuda:
        model = model.cuda()

    # Load the batch size parameter from the network/model
    batch_size = networks['batch_size']

    # Update the model's state dictionary to the one provided by the user.
    model.load_state_dict(checkpoint['model_state_dict'])

    # Need to extract the values used to normalizing
    # the training data provided to VAEAC during training.
    # Thus, the new data should have the same underlying
    # structure for VAEAC to compute reasonable imputations.
    norm_mean = checkpoint['norm_mean']
    norm_std = checkpoint['norm_std']

    # Then we need to let the model know if we are going to
    # continue training or just evaluate instances.
    model.eval()

    if verbose:
        print("\nLoaded the VAEAC model.", file=stderr, flush=True)

    # Convert the provided input data to torch
    # convert first to np to ensure that workflow from R also works when
    # we are only imputing for a single instance. I.e., one row.
    # Had some issues with it converting to list instead of array
    # without calling np.array first.
    instances_to_impute_torch = torch.from_numpy(np.array(instances_to_impute)).float()

    # Normalize the data to have mean = 0 and std = 1.
    # Use [None] to transforms vector to matrix
    data_nan = (instances_to_impute_torch - norm_mean[None]) / norm_std[None]

    # Here the imputations start
    if verbose:
        print("Start preparation work before imputations.", file=stderr, flush=True)

    # Non-zero number of workers cause nasty warnings because of some bug in
    # multiprocess library. It might be fixed now, but anyway there is no need
    # to have a lot of workers for dataloader over in-memory tabular data.
    num_workers = 0

    # Create a dataloader to load the data. Do not need to
    # randomize the reading order as all instances are
    # going to be sent through the VAEAC model.
    dataloader = DataLoader(data_nan,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False)

    # Start preparations for storing the imputations.
    # Results will hold the imputed versions, where each
    # new imputation is a list/array inside results
    results_prior_loc = []
    results_prior_scale = []
    results_proposal_loc = []
    results_proposal_scale = []

    # Rename the dataloader to iterator as convention
    iterator = dataloader

    # Create a progress bar that shows the progress of imputations
    if verbose:
        print("Ready to start imputing the instances.", file=stderr, flush=True)
        iterator = tqdm(iterator)

    # Impute missing values for all input data
    # Take one batch (64 instances) at the time.
    for batch in iterator:

        # Creates a deep copy
        # batch_extended = torch.tensor(batch)
        batch_extended = batch.clone().detach()

        # if batch size is less than batch_size, extend it with objects
        # from the beginning of the dataset
        batch_extended = extend_batch(batch_extended, dataloader, batch_size)

        # Send the original and extended batch to GPU if applicable
        if use_cuda:
            batch = batch.cuda()
            batch_extended = batch_extended.cuda()

        # Compute the imputation mask
        mask_extended = torch.isnan(batch_extended).float()

        # Compute imputation distributions parameters
        # Do not need to keep track of the gradients, as we are not fitting the networks
        with torch.no_grad():
            # Send the extended batch through the proposal and prior network.
            # For each, we get independent normal distribution of dimension equal
            # to 'latent_dim', where each dimension is independent.
            proposal, prior = model.make_latent_distributions(batch_extended, mask_extended)

            # Extract the obtained location matrix and scale matrix for the
            # proposal network. They are of size batch_size x latent_dim,
            # but we only want the results for the bach and not batch_extended
            proposal_loc = proposal.loc.clone().detach().cpu().numpy()[:batch.shape[0]]
            proposal_scale = proposal.scale.clone().detach().cpu().numpy()[:batch.shape[0]]

            # Extract the obtained location matrix and scale matrix for the
            # proposal network. They are of size batch_size x latent_dim,
            # but we only want the results for the bach and not batch_extended.
            prior_loc = prior.loc.clone().detach().cpu().numpy()[:batch.shape[0]]
            prior_scale = prior.scale.clone().detach().cpu().numpy()[:batch.shape[0]]

        # Add the different locations and scales to the corresponding result list
        results_prior_loc.append(prior_loc)
        results_prior_scale.append(prior_scale)
        results_proposal_loc.append(proposal_loc)
        results_proposal_scale.append(proposal_scale)

    # Concatenate the list of matrices into one large matrix, for each setting
    results_prior_loc_matrix = np.concatenate(results_prior_loc)
    results_prior_scale_matrix = np.concatenate(results_prior_scale)
    results_proposal_loc_matrix = np.concatenate(results_proposal_loc)
    results_proposal_scale_matrix = np.concatenate(results_proposal_scale)

    # Creat a dictionary with the results
    sub_dict = {"results_prior_loc": results_prior_loc_matrix,
                "results_prior_scale": results_prior_scale_matrix,
                "results_proposal_loc": results_proposal_loc_matrix,
                "results_proposal_scale": results_proposal_scale_matrix}

    return sub_dict





def VAEAC_impute_values(instances_to_impute,
                        path_VAEAC_model,
                        num_imputations,
                        use_cuda,
                        one_hot_max_sizes,
                        use_skip_connections=True,
                        verbose=False
                        ):
    """
    #os.chdir("/Users/larsolsen/Desktop/PhD/VAEAC_Updated_Version/")
    instances_to_impute = [[float('nan'), float('nan'), float('nan')],
                           [0.0056469140, float('nan'), float('nan')],
                           [float('nan'), -0.028083610, float('nan')],
                           [0.0056469140, -0.028083610, float('nan')],
                           [float('nan'), float('nan'), -0.653991],
                           [0.0056469140, float('nan'), -0.653991],
                           [float('nan'), -0.028083610, -0.653991],
                           [0.0056469140, -0.028083610, -0.653991]]
    one_hot_max_sizes = [1, 1, 1]
    use_cuda = False
    verbose = False
    num_imputations = 10
    path_VAEAC_model = "/Users/larsolsen/PhD/Paper1/FittedModels/Normal/normal_p_3_rho_080_n_2000_best.pt"
    """

    """
    Function that imputes the missing values in 2D matrix where each row constitute
    an individual. The values are sampled from the conditional distribution
    estimated by a VAEAC model.

    instances_to_impute: a 2^p (maybe 2^p-2) x p matrix in R. Gets automatically
                         transformed to a numpy.ndarray when sent to python.
                         Should be 'nan' (or 'NaN' in R) where we want to impute
                         the missing values.
    path_VAEAC_model: string containing the location of the saved VAEAC model
    num_imputations: the number of imputed versions we create for each row in 'instances_to_impute'
    one_hot_max_sizes: array with the one-hot max sizes for categorical features and 0 or 1
                       for real-valued ones. We deal only with real-valued. The length of
                       the array must be equal to the number of columns/dimension of covariates.


    """

    # If we are asked to run on cuda, check if it is possible
    cuda_available = torch.cuda.is_available()

    # Give warning to user if asked to run on cuda, but cuda is not available.
    if cuda_available is False and use_cuda is True:
        warnings.warn("Cuda is not available. Fall back to CPU.", ResourceWarning)

    # Define boolean whether we use cuda or not, then use CPU.
    use_cuda = cuda_available and use_cuda

    # Load the VAEAC model at the provided path.
    # This loads a dictionary that contains the following elements (OLD VERSION):
    # 'epoch', 'model_state_dict', 'optimizer_state_dict',
    # 'validation_iwae', 'train_vlb', 'norm_mean', and 'norm_std'.
    #
    # IN THE NEW VERSIONS, THE DICTIONARY CONTAINS:
    # 'epoch', 'model_state_dict', 'optimizer_state_dict',
    # 'validation_iwae', 'validation_iwae_running_avg',
    # 'running_avg_num_values', 'train_vlb',
    # 'norm_mean', 'norm_std', 'distribution', 'n', 'p',
    # 'param_now', 'one_hot_max_sizes', 'epochs', 'masking_ratio',
    # 'validation_ratio', 'validation_iwae_num_samples',
    # 'validations_per_epoch', 'num_different_vaeac_initiate',
    # 'epochs_initiation_phase', 'width', 'depth',
    # 'latent_dim', 'lr', and 'batch_size'.
    # SO I DO NOT NEED TO DO THE THINGS BELOW.
    # AND THIS FUNCTION CAN TAKE IN LESS PARAMETERS
    checkpoint = torch.load(path_VAEAC_model)
    # checkpoint = torch.load("/Users/larsolsen/PhD/Paper1/ShapleyValuesBurr/VAEAC_models/burr_ntrain_100_repetition_19_width_32_depth_3_k_1000_lr_7e-04_without_skip_p_10_param_2.0_n_100_depth_3_width_32_latent_8_lr_0.0007_best.pt")

    if use_skip_connections:
        # Extract some of the parameters based on the dimensions of the networks
        depth = int(list(checkpoint['model_state_dict'].keys())[-1].split(".")[1]) - 4
        width, latent_dim = checkpoint['model_state_dict']['generative_network.0.weight'].shape
        lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']

        # Design all necessary networks and learning parameters for the dataset
        # Might be possible to eliminate some stuff here to make it faster
        # as we are not going to continue training.
        networks = get_imputation_networks(one_hot_max_sizes=one_hot_max_sizes,
                                           width=width,
                                           depth=depth,
                                           latent_dim=latent_dim,
                                           lr=lr)
    else:
        # Extract some of the parameters based on the dimensions of the networks
        depth = int(list(checkpoint['model_state_dict'].keys())[-1].split(".")[1]) - 5
        width, latent_dim = checkpoint['model_state_dict']['generative_network.0.weight'].shape
        lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']

        # Design all necessary networks and learning parameters for the dataset
        # Might be possible to eliminate some stuff here to make it faster
        # as we are not going to continue training.
        networks = get_imputation_networks_without_skip_connections(
            one_hot_max_sizes=one_hot_max_sizes,
            width=width,
            depth=depth,
            latent_dim=latent_dim,
            lr=lr)

    # Create the VAEAC model by providing the necessary networks
    model = VAEAC(
        networks['reconstruction_log_prob'],
        networks['proposal_network'],
        networks['prior_network'],
        networks['generative_network']
    )

    # Send the model to the GPU, if we are supposed to.
    if use_cuda:
        model = model.cuda()

    # Load the batch size parameter from the network/model
    batch_size = networks['batch_size']

    # Create the optimizer used to compute the updated in
    # back propagation.
    # THIS CAN BE REMOVED AS WE ARE NOT TRAINING THE NETWORK.
    # optimizer = networks['optimizer'](model.parameters())

    # THIS CAN BE REMOVED AS WE ARE NOT TRAINING THE NETWORK.
    # mask_generator = networks['mask_generator']
    # vlb_scale_factor = networks.get('vlb_scale_factor', 1)

    # Update the model's state dictionary to the one provided by the user.
    model.load_state_dict(checkpoint['model_state_dict'])

    # To continue the training we also need to update the optimiser's,
    # however, as we only use the VAEAC model we can skip this.
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Need to extract the values used to normalizing
    # the training data provided to VAEAC during training.
    # Thus, the new data should have the same underlying
    # structure for VAEAC to compute reasonable imputations.
    norm_mean = checkpoint['norm_mean']
    norm_std = checkpoint['norm_std']

    # Then we need to let the model know if we are going to
    # continue training or just evaluate instances.
    # model.train()
    #  - or -
    model.eval()

    if verbose:
        print("\nLoaded the VAEAC model.", file=stderr, flush=True)

    """
    # Data which I used for testing
    instances_to_impute2 = [[float('nan'), float('nan'), float('nan')],
                           [0.0056469140, float('nan'), float('nan')],
                           [float('nan'), -0.028083610, float('nan')],
                           [0.0056469140, -0.028083610, float('nan')],
                           [float('nan'), float('nan'), -0.653991],
                           [0.0056469140, float('nan'), -0.653991],
                           [float('nan'), -0.028083610, -0.653991],
                           [0.0056469140, -0.028083610, -0.653991]]
    """

    # Convert the provided input data to torch
    # convert first to np to ensure that workflow from R also works when
    # we are only imputing for a single instance. I.e., one row.
    # Had some issues with it converting to list instead of array
    # without calling np.array first.
    instances_to_impute_torch = torch.from_numpy(np.array(instances_to_impute)).float()

    # Normalize the data to have mean = 0 and std = 1.
    # Use [None] to transforms vector to matrix
    data_nan = (instances_to_impute_torch - norm_mean[None]) / norm_std[None]

    # Here the imputations start
    if verbose:
        print("Start preparation work before imputations.", file=stderr, flush=True)

    # Non-zero number of workers cause nasty warnings because of some bug in
    # multiprocess library. It might be fixed now, but anyway there is no need
    # to have a lot of workers for dataloader over in-memory tabular data.
    num_workers = 0

    # Create a dataloader to load the data. Do not need to
    # randomize the reading order as all instances are
    # going to be sent through the VAEAC model.
    dataloader = DataLoader(data_nan,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False)

    # Start preparations for storing the imputations.
    # Results will hold the imputed versions, where each
    # new imputation is a list/array inside results
    results = []

    # We add num_imputations number of empty lists to the results array
    for i in range(num_imputations):
        results.append([])

    # Rename the dataloader to iterator as convention
    iterator = dataloader

    # Create a progress bar that shows the progress of imputations
    if verbose:
        print("Ready to start imputing the instances.", file=stderr, flush=True)
        iterator = tqdm(iterator)

    # Impute missing values for all input data
    # Take one batch (64 instances) at the time.
    for batch in iterator:

        # Creates a deep copy
        # batch_extended = torch.tensor(batch)
        batch_extended = batch.clone().detach()

        # if batch size is less than batch_size, extend it with objects
        # from the beginning of the dataset
        batch_extended = extend_batch(batch_extended, dataloader, batch_size)

        # Send the original and extended batch to GPU if applicable
        if use_cuda:
            batch = batch.cuda()
            batch_extended = batch_extended.cuda()

        # Compute the imputation mask
        mask_extended = torch.isnan(batch_extended).float()

        # Compute imputation distributions parameters
        # Do not need to keep track of the gradients, as we are not fitting the networks
        with torch.no_grad():
            # We here get a tensor of size (batch_size, K, 2*num_features)
            # Where the last dimension are the mu and sigma for each
            # feature which is assumed to be normal.
            # Will be different when we also include categorical.
            # Then, for each instance in batch, we get num_imputations versions.
            samples_params = model.generate_samples_params(batch_extended,
                                                           mask_extended,
                                                           num_imputations)

            # We only want the parameters of the values in batch_size and not
            # the added instances in batch_extended. So remove those
            # This indexing is equivalent to [:batch.shape[0], :, :]
            samples_params = samples_params[:batch.shape[0]]

        # Make a deep copy of batch with missing values set to zero.
        mask = torch.isnan(batch)
        # batch_zeroed_nans = torch.tensor(batch)
        batch_zeroed_nans = batch.clone().detach()
        batch_zeroed_nans[mask] = 0

        # Impute samples from the generative distributions into the data
        # and save it to the results
        for i in range(num_imputations):
            # Get the i-th imputed parameters for all the instances in the batch
            # for samples_params is a tensor of size
            # (batch_size, num_imputations, 2*num_features)
            # so sample_params is of size (batch_size, 2*num_features)
            # A bit different for categorical features as we
            # there do not have mu and sigma for generative normal at the end.
            sample_params = samples_params[:, i]

            # Use a Gaussian sampler to create the final imputed values.
            # For the continuous case, this corresponds to returning
            # the means for each feature obtained in sample_params.
            # But we use random sampling at the end, so we DO NOT use
            # the mean but rather sample from the distribution.
            # A bit different for categorical features as we do not use
            # GAUSSIAN sampler but rather CATEGORICAL sampler.
            sample = networks['sampler_most_prob_false'](sample_params)

            # We do not want to keep the imputed values for those
            # features that we know the true value. So we set all
            # the sampled values for known features to zero.
            sample[torch.logical_not(mask)] = 0

            # Then we combine the sampled values with the batch
            # so that the imputed values are added to only the
            # missing values. Sample is size (batch_size, num_features)
            sample += batch_zeroed_nans

            # Add the a copy of the i'th sampled values
            # to the i'th list inside the results list
            results[i].append(sample.clone().detach().cpu())
            # results[i].append(torch.tensor(sample, device='cpu'))

    if verbose:
        print("The imputations are done. Start concatenating the results.",
              file=stderr, flush=True)

    # Concatenate all batches into one [n x K x D] tensor,
    # where n in the number of objects, K is the number of imputations
    # and D is the dimensionality of one object
    # iterate over num_imputations
    for i in range(len(results)):
        # Make a copy of the i'th list in results, which is of n x num_features
        # then unsqueeze to get size n x 1 x num_features.
        # overwrite the ith list in results such that
        # results[i] is matrix of dim  n x 1 x num_features.
        results[i] = torch.cat(results[i]).unsqueeze(1)

    # Then concatenate the list of matrices of dim: n x 1 x num_features,
    # around the second axis so we get a tensor of dim: n x num_imputations x num_features
    result = torch.cat(results, 1)

    # reshape result, undo normalization and save it
    # Reshape it such that it is a two dim matrix of
    # dim: (n x num_imputations) x num_features
    # So first num_imputations rows are the imputed versions of the first instance
    # This might not be needed here
    # result = result.view(result.shape[0] * result.shape[1], result.shape[2])

    # Undo the normalization (data - mu)/sigma by
    # multiplying by sigma and adding the mean
    result = result * norm_std[None] + norm_mean[None]

    # Return the input data but now with the missing values imputed
    # based on the conditional distributions estimated by VAEAC.
    # The return object will be read as a three dimensional array in R with dimension
    # nrow(instances_to_impute) x num_imputations x ncol(instances_to_impute)
    # The latter is also the dimension of the data
    return result.detach().cpu().numpy()

# %%
def VAEAC_impute_values_return_most_likely(instances_to_impute,
                                           path_VAEAC_model,
                                           num_imputations,
                                           use_cuda,
                                           one_hot_max_sizes,
                                           use_skip_connections=True,
                                           verbose=False
                                           ):
    """
    #os.chdir("/Users/larsolsen/Desktop/PhD/VAEAC_Updated_Version/")
    instances_to_impute = [[float('nan'), float('nan'), float('nan')],
                           [0.0056469140, float('nan'), float('nan')],
                           [float('nan'), -0.028083610, float('nan')],
                           [0.0056469140, -0.028083610, float('nan')],
                           [float('nan'), float('nan'), -0.653991],
                           [0.0056469140, float('nan'), -0.653991],
                           [float('nan'), -0.028083610, -0.653991],
                           [0.0056469140, -0.028083610, -0.653991]]
    one_hot_max_sizes = [1, 1, 1]
    use_cuda = False
    verbose = False
    num_imputations = 10
    path_VAEAC_model = "/Users/larsolsen/PhD/Paper1/FittedModels/Normal/normal_p_3_rho_080_n_2000_best.pt"
    """

    """
    Function that imputes the missing values in 2D matrix where each row constitute
    an individual. The values are sampled from the conditional distribution
    estimated by a VAEAC model.

    instances_to_impute: a 2^p (maybe 2^p-2) x p matrix in R. Gets automatically
                         transformed to a numpy.ndarray when sent to python.
                         Should be 'nan' (or 'NaN' in R) where we want to impute
                         the missing values.
    path_VAEAC_model: string containing the location of the saved VAEAC model
    num_imputations: the number of imputed versions we create for each row in 'instances_to_impute'
    one_hot_max_sizes: array with the one-hot max sizes for categorical features and 0 or 1
                       for real-valued ones. We deal only with real-valued. The length of
                       the array must be equal to the number of columns/dimension of covariates.


    """
    # %%
    # If we are asked to run on cuda, check if it is possible
    cuda_available = torch.cuda.is_available()

    # Give warning to user if asked to run on cuda, but cuda is not available.
    if cuda_available is False and use_cuda is True:
        warnings.warn("Cuda is not available. Fall back to CPU.", ResourceWarning)

    # Define boolean whether we use cuda or not, then use CPU.
    use_cuda = cuda_available and use_cuda

    # Design all necessary networks and learning parameters for the dataset
    # Might be possible to eliminate some stuff here to make it faster
    # as we are not going to continue training.
    if use_skip_connections:
        networks = get_imputation_networks(one_hot_max_sizes)
    else:
        networks = get_imputation_networks_without_skip_connections(one_hot_max_sizes)

    # Crate the VAEAC model by providing the necessary networks
    model = VAEAC(
        networks['reconstruction_log_prob'],
        networks['proposal_network'],
        networks['prior_network'],
        networks['generative_network']
    )

    # Send the model to the GPU, if we are supposed to.
    if use_cuda:
        model = model.cuda()

    # Load the batch size parameter from the network/model
    batch_size = networks['batch_size']

    # Create the optimizer used to compute the updated in
    # back propagation.
    # THIS CAN BE REMOVED AS WE ARE NOT TRAINING THE NETWORK.
    # optimizer = networks['optimizer'](model.parameters())

    # THIS CAN BE REMOVED AS WE ARE NOT TRAINING THE NETWORK.
    # mask_generator = networks['mask_generator']
    # vlb_scale_factor = networks.get('vlb_scale_factor', 1)

    # Load the VAEAC model at the provided path.
    # This loads a dictionary that contains the following elements:
    # 'epoch', 'model_state_dict', 'optimizer_state_dict',
    # 'validation_iwae', 'train_vlb', 'norm_mean', and 'norm_std'.
    checkpoint = torch.load(path_VAEAC_model)

    # Update the model's state dictionary to the one provided by the user.
    model.load_state_dict(checkpoint['model_state_dict'])

    # To continue the training we also need to update the optimiser's,
    # however, as we only use the VAEAC model we can skip this.
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Need to extract the values used to normalizing
    # the training data provided to VAEAC during training.
    # Thus, the new data should have the same underlying
    # structure for VAEAC to compute reasonable imputations.
    norm_mean = checkpoint['norm_mean']
    norm_std = checkpoint['norm_std']

    # Then we need to let the model know if we are going to
    # continue training or just evaluate instances.
    # model.train()
    #  - or -
    model.eval()

    if verbose:
        print("\nLoaded the VAEAC model.", file=stderr, flush=True)

    """
    # Data which I used for testing
    instances_to_impute2 = [[float('nan'), float('nan'), float('nan')],
                           [0.0056469140, float('nan'), float('nan')],
                           [float('nan'), -0.028083610, float('nan')],
                           [0.0056469140, -0.028083610, float('nan')],
                           [float('nan'), float('nan'), -0.653991],
                           [0.0056469140, float('nan'), -0.653991],
                           [float('nan'), -0.028083610, -0.653991],
                           [0.0056469140, -0.028083610, -0.653991]]
    """

    # Convert the provided input data to torch
    # convert first to np to ensure that workflow from R also works when
    # we are only imputing for a single instance. I.e., one row.
    # Had some issues with it converting to list instead of array
    # without calling np.array first.
    instances_to_impute_torch = torch.from_numpy(np.array(instances_to_impute)).float()

    # Normalize the data to have mean = 0 and std = 1.
    # Use [None] to transforms vector to matrix
    data_nan = (instances_to_impute_torch - norm_mean[None]) / norm_std[None]

    # Here the imputations start
    if verbose:
        print("Start preparation work before imputations.", file=stderr, flush=True)

    # Non-zero number of workers cause nasty warnings because of some bug in
    # multiprocess library. It might be fixed now, but anyway there is no need
    # to have a lot of workers for dataloader over in-memory tabular data.
    num_workers = 0

    # Create a dataloader to load the data. Do not need to
    # randomize the reading order as all instances are
    # going to be sent through the VAEAC model.
    dataloader = DataLoader(data_nan,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False)

    # Start preparations for storing the imputations.
    # Results will hold the imputed versions, where each
    # new imputation is a list/array inside results
    results = []

    # We add num_imputations number of empty lists to the results array
    for i in range(num_imputations):
        results.append([])

    # Rename the dataloader to iterator as convention
    iterator = dataloader

    # Create a progress bar that shows the progress of imputations
    if verbose:
        print("Ready to start imputing the instances.", file=stderr, flush=True)
        iterator = tqdm(iterator)

    # %%
    # Impute missing values for all input data
    # Take one batch (64 instances) at the time.
    for batch in iterator:

        # Creates a deep copy
        # batch_extended = torch.tensor(batch)
        batch_extended = batch.clone().detach()

        # if batch size is less than batch_size, extend it with objects
        # from the beginning of the dataset
        batch_extended = extend_batch(batch_extended, dataloader, batch_size)

        # Send the original and extended batch to GPU if applicable
        if use_cuda:
            batch = batch.cuda()
            batch_extended = batch_extended.cuda()

        # Compute the imputation mask
        mask_extended = torch.isnan(batch_extended).float()

        # Compute imputation distributions parameters
        # Do not need to keep track of the gradients, as we are not fitting the networks
        with torch.no_grad():
            # We here get a tensor of size (batch_size, K, 2*num_features)
            # Where the last dimension are the mu and sigma for each
            # feature which is assumed to be normal.
            # Then, for each instance in batch, we get num_imputations versions.
            samples_params = model.generate_samples_params(batch_extended,
                                                           mask_extended,
                                                           num_imputations)

            # We only want the parameters of the values in batch_size and not
            # the added instances in batch_extended. So remove those
            # This indexing is equivalent to [:batch.shape[0], :, :]
            samples_params = samples_params[:batch.shape[0]]

        # Make a deep copy of batch with missing values set to zero.
        mask = torch.isnan(batch)
        # batch_zeroed_nans = torch.tensor(batch)
        batch_zeroed_nans = batch.clone().detach()
        batch_zeroed_nans[mask] = 0

        # Impute samples from the generative distributions into the data
        # and save it to the results
        for i in range(num_imputations):
            # Get the i-th imputed parameters for all the instances in the batch
            # for samples_params is a tensor of size
            # (batch_size, num_imputations, 2*num_features)
            # so sample_params is of size (batch_size, 2*num_features)
            sample_params = samples_params[:, i]

            # Use a Gaussian sampler to create the final imputed values.
            # For the continuous case, this corresponds to returning
            # the means for each feature obtained in sample_params.
            # But we use random sampling at the end, so we DO NOT use
            # the mean but rather sample from the distribution.
            sample = networks['sampler_most_prob_true'](sample_params)

            # We do not want to keep the imputed values for those
            # features that we know the true value. So we set all
            # tha sampled values for known features to zero.
            sample[torch.logical_not(mask)] = 0

            # Then we combine the sampled values with the batch
            # so that the imputed values are added to only the
            # missing values. Sample is size (batch_size, num_features)
            sample += batch_zeroed_nans

            # Add the a copy of the i'th sampled values
            # to the i'th list inside the results list
            results[i].append(sample.clone().detach().cpu())
            # results[i].append(torch.tensor(sample, device='cpu'))

    if verbose:
        print("The imputations are done. Start concatenating the results.",
              file=stderr, flush=True)

    # Concatenate all batches into one [n x K x D] tensor,
    # where n in the number of objects, K is the number of imputations
    # and D is the dimensionality of one object
    # iterate over num_imputations
    for i in range(len(results)):
        # Make a copy of the i'th list in results, which is of n x num_features
        # then unsqueeze to get size n x 1 x num_features.
        # overwrite the ith list in results such that
        # results[i] is matrix of dim  n x 1 x num_features.
        results[i] = torch.cat(results[i]).unsqueeze(1)

    # Then concatenate the list of matrices of dim: n x 1 x num_features,
    # around the second axis so we get a tensor of dim: n x num_imputations x num_features
    result = torch.cat(results, 1)

    # reshape result, undo normalization and save it
    # Reshape it such that it is a two dim matrix of
    # dim: (n x num_imputations) x num_features
    # So first num_imputations rows are the imputed versions of the first instance
    # This might not be needed here
    # result = result.view(result.shape[0] * result.shape[1], result.shape[2])

    # Undo the normalization (data - mu)/sigma by
    # multiplying by sigma and adding the mean
    result = result * norm_std[None] + norm_mean[None]

    # Return the input data but now with the missing values imputed
    # based on the conditional distributions estimated by VAEAC.
    # The return object will be read as a three dimensional array in R with dimension
    # nrow(instances_to_impute) x num_imputations x ncol(instances_to_impute)
    # The latter is also the dimension of the data
    return result.detach().cpu().numpy()

# %%
def VAEAC_impute_means_and_variances(instances_to_impute,
                        path_VAEAC_model,
                        num_imputations,
                        use_cuda,
                        one_hot_max_sizes,
                        use_skip_connections=True,
                        verbose=False
                        ):
    """
    os.chdir("/Users/larsolsen/Desktop/PhD/VAEAC_Updated_Version/")
    instances_to_impute = [[float('nan'), float('nan'), float('nan')],
                           [0.0056469140, float('nan'), float('nan')],
                           [float('nan'), -0.028083610, float('nan')],
                           [0.0056469140, -0.028083610, float('nan')],
                           [float('nan'), float('nan'), -0.653991],
                           [0.0056469140, float('nan'), -0.653991],
                           [float('nan'), -0.028083610, -0.653991],
                           [0.0056469140, -0.028083610, -0.653991]]
    one_hot_max_sizes = [1, 1, 1]
    use_cuda = False
    verbose = False
    num_imputations = 10
    path_VAEAC_model = "/Users/larsolsen/PhD/Paper1/FittedModels/Normal/normal_p_3_rho_080_n_2000_best.pt"
    """

    """
    Function that imputes the missing values in 2D matrix where each row constitute
    an individual. The values are sampled from the conditional distribution
    estimated by a VAEAC model.

    instances_to_impute: a 2^p (maybe 2^p-2) x p matrix in R. Gets automatically
                         transformed to a numpy.ndarray when sent to python.
                         Should be 'nan' (or 'NaN' in R) where we want to impute
                         the missing values.
    path_VAEAC_model: string containing the location of the saved VAEAC model
    num_imputations: the number of imputed versions we create for each row in 'instances_to_impute'
    one_hot_max_sizes: array with the one-hot max sizes for categorical features and 0 or 1
                       for real-valued ones. We deal only with real-valued. The length of
                       the array must be equal to the number of columns/dimension of covariates.


    """
    # %%
    # If we are asked to run on cuda, check if it is possible
    cuda_available = torch.cuda.is_available()

    # Give warning to user if asked to run on cuda, but cuda is not available.
    if cuda_available is False and use_cuda is True:
        warnings.warn("Cuda is not available. Fall back to CPU.", ResourceWarning)

    # Define boolean whether we use cuda or not, then use CPU.
    use_cuda = cuda_available and use_cuda

    # Load the VAEAC model at the provided path.
    # This loads a dictionary that contains the following elements (OLD VERSION):
    # 'epoch', 'model_state_dict', 'optimizer_state_dict',
    # 'validation_iwae', 'train_vlb', 'norm_mean', and 'norm_std'.
    #
    # IN THE NEW VERSIONS, THE DICTIONARY CONTAINS:
    # 'epoch', 'model_state_dict', 'optimizer_state_dict',
    # 'validation_iwae', 'validation_iwae_running_avg',
    # 'running_avg_num_values', 'train_vlb',
    # 'norm_mean', 'norm_std', 'distribution', 'n', 'p',
    # 'param_now', 'one_hot_max_sizes', 'epochs', 'masking_ratio',
    # 'validation_ratio', 'validation_iwae_num_samples',
    # 'validations_per_epoch', 'num_different_vaeac_initiate',
    # 'epochs_initiation_phase', 'width', 'depth',
    # 'latent_dim', 'lr', and 'batch_size'.
    # SO I DO NOT NEED TO DO THE THINGS BELOW.
    # AND THIS FUNCTION CAN TAKE IN LESS PARAMETERS
    checkpoint = torch.load(path_VAEAC_model)
    # checkpoint = torch.load("/Users/larsolsen/PhD/Paper1/ShapleyValuesBurr/VAEAC_models/burr_ntrain_100_repetition_19_width_32_depth_3_k_1000_lr_7e-04_without_skip_p_10_param_2.0_n_100_depth_3_width_32_latent_8_lr_0.0007_best.pt")

    if use_skip_connections:
        # Extract some of the parameters based on the dimensions of the networks
        depth = int(list(checkpoint['model_state_dict'].keys())[-1].split(".")[1]) - 4
        width, latent_dim = checkpoint['model_state_dict']['generative_network.0.weight'].shape
        lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']

        # Design all necessary networks and learning parameters for the dataset
        # Might be possible to eliminate some stuff here to make it faster
        # as we are not going to continue training.
        networks = get_imputation_networks(one_hot_max_sizes=one_hot_max_sizes,
                                           width=width,
                                           depth=depth,
                                           latent_dim=latent_dim,
                                           lr=lr)
    else:
        # Extract some of the parameters based on the dimensions of the networks
        depth = int(list(checkpoint['model_state_dict'].keys())[-1].split(".")[1]) - 5
        width, latent_dim = checkpoint['model_state_dict']['generative_network.0.weight'].shape
        lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']

        # Design all necessary networks and learning parameters for the dataset
        # Might be possible to eliminate some stuff here to make it faster
        # as we are not going to continue training.
        networks = get_imputation_networks_without_skip_connections(
            one_hot_max_sizes=one_hot_max_sizes,
            width=width,
            depth=depth,
            latent_dim=latent_dim,
            lr=lr)

    # Create the VAEAC model by providing the necessary networks
    model = VAEAC(
        networks['reconstruction_log_prob'],
        networks['proposal_network'],
        networks['prior_network'],
        networks['generative_network']
    )

    # Send the model to the GPU, if we are supposed to.
    if use_cuda:
        model = model.cuda()

    # Load the batch size parameter from the network/model
    batch_size = networks['batch_size']

    # Create the optimizer used to compute the updated in
    # back propagation.
    # THIS CAN BE REMOVED AS WE ARE NOT TRAINING THE NETWORK.
    # optimizer = networks['optimizer'](model.parameters())

    # THIS CAN BE REMOVED AS WE ARE NOT TRAINING THE NETWORK.
    # mask_generator = networks['mask_generator']
    # vlb_scale_factor = networks.get('vlb_scale_factor', 1)

    # Update the model's state dictionary to the one provided by the user.
    model.load_state_dict(checkpoint['model_state_dict'])

    # To continue the training we also need to update the optimiser's,
    # however, as we only use the VAEAC model we can skip this.
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Need to extract the values used to normalizing
    # the training data provided to VAEAC during training.
    # Thus, the new data should have the same underlying
    # structure for VAEAC to compute reasonable imputations.
    norm_mean = checkpoint['norm_mean']
    norm_std = checkpoint['norm_std']

    # Then we need to let the model know if we are going to
    # continue training or just evaluate instances.
    # model.train()
    #  - or -
    model.eval()

    if verbose:
        print("\nLoaded the VAEAC model.", file=stderr, flush=True)

    """
    # Data which I used for testing
    instances_to_impute2 = [[float('nan'), float('nan'), float('nan')],
                           [0.0056469140, float('nan'), float('nan')],
                           [float('nan'), -0.028083610, float('nan')],
                           [0.0056469140, -0.028083610, float('nan')],
                           [float('nan'), float('nan'), -0.653991],
                           [0.0056469140, float('nan'), -0.653991],
                           [float('nan'), -0.028083610, -0.653991],
                           [0.0056469140, -0.028083610, -0.653991]]
    """

    # Convert the provided input data to torch
    # convert first to np to ensure that workflow from R also works when
    # we are only imputing for a single instance. I.e., one row.
    # Had some issues with it converting to list instead of array
    # without calling np.array first.
    instances_to_impute_torch = torch.from_numpy(np.array(instances_to_impute)).float()

    # Normalize the data to have mean = 0 and std = 1.
    # Use [None] to transforms vector to matrix
    data_nan = (instances_to_impute_torch - norm_mean[None]) / norm_std[None]

    # Here the imputations start
    if verbose:
        print("Start preparation work before imputations.", file=stderr, flush=True)

    # Non-zero number of workers cause nasty warnings because of some bug in
    # multiprocess library. It might be fixed now, but anyway there is no need
    # to have a lot of workers for dataloader over in-memory tabular data.
    num_workers = 0

    # Create a dataloader to load the data. Do not need to
    # randomize the reading order as all instances are
    # going to be sent through the VAEAC model.
    dataloader = DataLoader(data_nan,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False)

    # Start preparations for storing the imputations.
    # Results will hold the imputed versions, where each
    # new imputation is a list/array inside results
    results = []
    results_means = []
    results_sigmas = []

    # We add num_imputations number of empty lists to the results array
    for i in range(num_imputations):
        results.append([])
        results_means.append([])
        results_sigmas.append([])

    # Rename the dataloader to iterator as convention
    iterator = dataloader

    # Create a progress bar that shows the progress of imputations
    if verbose:
        print("Ready to start imputing the instances.", file=stderr, flush=True)
        iterator = tqdm(iterator)
    # %%
    # Impute missing values for all input data
    # Take one batch (64 instances) at the time.
    for batch in iterator:

        # %%
        # Creates a deep copy
        # batch_extended = torch.tensor(batch)
        batch_extended = batch.clone().detach()

        # if batch size is less than batch_size, extend it with objects
        # from the beginning of the dataset
        batch_extended = extend_batch(batch_extended, dataloader, batch_size)

        # Send the original and extended batch to GPU if applicable
        if use_cuda:
            batch = batch.cuda()
            batch_extended = batch_extended.cuda()

        # Compute the imputation mask
        mask_extended = torch.isnan(batch_extended).float()

        # Compute imputation distributions parameters
        # Do not need to keep track of the gradients, as we are not fitting the networks
        with torch.no_grad():
            # We here get a tensor of size (batch_size, K, 2*num_features)
            # Where the last dimension are the mu and sigma for each
            # feature which is assumed to be normal.
            # Will be different when we also include categorical.
            # Then, for each instance in batch, we get num_imputations versions.
            samples_params = model.generate_samples_params(batch_extended,
                                                           mask_extended,
                                                           num_imputations)

            # We only want the parameters of the values in batch_size and not
            # the added instances in batch_extended. So remove those
            # This indexing is equivalent to [:batch.shape[0], :, :]
            samples_params = samples_params[:batch.shape[0]]

        # Know samples_params.size() = [batch_size, num_imputations, 2*M],
        # Where M is the number of features. Only valid for continuous data.

        # %%
        # Make a deep copy of batch with missing values set to zero.
        mask = torch.isnan(batch)
        # batch_zeroed_nans = torch.tensor(batch)
        batch_zeroed_nans = batch.clone().detach()
        batch_zeroed_nans[mask] = 0

        # %%
        # Impute samples from the generative distributions into the data
        # and save it to the results
        for i in range(num_imputations):
            # Get the i-th imputed parameters for all the instances in the batch
            # for samples_params is a tensor of size
            # (batch_size, num_imputations, 2*num_features)
            # so sample_params is of size (batch_size, 2*num_features)
            # A bit different for categorical features as we
            # there do not have mu and sigma for generative normal at the end.
            sample_params = samples_params[:, i]

            # Use a Gaussian sampler to create the final imputed values.
            # For the continuous case, this corresponds to returning
            # the means for each feature obtained in sample_params.
            # But we use random sampling at the end, so we DO NOT use
            # the mean but rather sample from the distribution.
            # A bit different for categorical features as we do not use
            # GAUSSIAN sampler but rather CATEGORICAL sampler.
            # sample = networks['sampler_most_prob_false'](sample_params)

            # A counter to keep track of which
            cur_distr_col = 0

            # List to store all the samples sampled from the
            # normal distribution with parameters from distr_params.
            sample_means = []
            sample_sigmas = []

            # %%
            # iterate over the number of features and the number
            # of categories for each feature.
            for j, size in enumerate(one_hot_max_sizes):

                # If we are dealing with continuous features
                if size <= 1:
                    # Gaussian distribution
                    # Get the mu and sigma for the current feature, for each instance
                    params = sample_params[:, cur_distr_col: cur_distr_col + 2]
                    cur_distr_col += 2

                    # Get the number of instances
                    n = params.shape[0]
                    # Then get the dimension of the parameters
                    d = params.shape[1]
                    # Use double dash to get integer. Do not need it as we by construction always have 2*num_dim_latent_space
                    mu = params[:, :d // 2]  # Get the first halves which are the means
                    sigma_params = params[:, d // 2:]  # Get the second half which are transformed sigmas
                    sigma = softplus(sigma_params)  # ln(1 + exp(sigma_params))
                    sigma = sigma.clamp(min=0.0001)

                    col_sigma = sigma
                    col_mean = mu
                else:
                    print("Does not work at the moment. Need to be implemented.")
                    # CATEGORICAL DO NOT SUPPORT THAT AT THE MOMENT

                # Add the vector of sampled values for the i´th
                # feature to the sample list.
                sample_means.append(col_mean)
                sample_sigmas.append(col_sigma)

            # %%
            sample_means = torch.cat(sample_means, 1)
            sample_sigmas = torch.cat(sample_sigmas, 1)

            # We do not want to keep the imputed values for those
            # features that we know the true value. So we set all
            # the sampled values for known features to zero.
            # sample[torch.logical_not(mask)] = 0

            # Then we combine the sampled values with the batch
            # so that the imputed values are added to only the
            # missing values. Sample is size (batch_size, num_features)
            # sample += batch_zeroed_nans

            # Add the a copy of the i'th sampled values
            # to the i'th list inside the results list
            # results[i].append(sample.clone().detach().cpu())
            # results[i].append(torch.tensor(sample, device='cpu'))
            results_means[i].append(sample_means.clone().detach().cpu())
            results_sigmas[i].append(sample_sigmas.clone().detach().cpu())

    # %%
    if verbose:
        print("The imputations are done. Start concatenating the results.",
              file=stderr, flush=True)

    # Concatenate all batches into one [n x K x D] tensor,
    # where n in the number of objects, K is the number of imputations
    # and D is the dimensionality of one object
    # iterate over num_imputations
    for i in range(len(results_sigmas)):
        # Make a copy of the i'th list in results, which is of n x num_features
        # then unsqueeze to get size n x 1 x num_features.
        # overwrite the ith list in results such that
        # results[i] is matrix of dim  n x 1 x num_features.
        # results[i] = torch.cat(results[i]).unsqueeze(1)
        results_means[i] = torch.cat(results_means[i]).unsqueeze(1)
        results_sigmas[i] = torch.cat(results_sigmas[i]).unsqueeze(1)

    # %%

    # Then concatenate the list of matrices of dim: n x 1 x num_features,
    # around the second axis so we get a tensor of dim: n x num_imputations x num_features
    # result = torch.cat(results, 1)
    result_means = torch.cat(results_means, 1)
    result_sigmas = torch.cat(results_sigmas, 1)

    # reshape result, undo normalization and save it
    # Reshape it such that it is a two dim matrix of
    # dim: (n x num_imputations) x num_features
    # So first num_imputations rows are the imputed versions of the first instance
    # This might not be needed here
    # result = result.view(result.shape[0] * result.shape[1], result.shape[2])

    # Undo the normalization (data - mu)/sigma by
    # multiplying by sigma and adding the mean
    # result = result * norm_std[None] + norm_mean[None]
    result_means = result_means * norm_std[None] + norm_mean[None]
    result_sigmas = result_sigmas * norm_std[None]

    # Return the input data but now with the missing values imputed
    # based on the conditional distributions estimated by VAEAC.
    # The return object will be read as a three dimensional array in R with dimension
    # nrow(instances_to_impute) x num_imputations x ncol(instances_to_impute)
    # The latter is also the dimension of the data
    # return result.detach().cpu().numpy()
    # %%
    return result_means.detach().cpu().numpy(), result_sigmas.detach().cpu().numpy()
