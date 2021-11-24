import torch
from tqdm import tqdm


def extend_batch(batch, dataloader, batch_size):
    """
    If the batch size is less than batch_size, extends it with
    data from the dataloader until it reaches the required size.
    Here batch is a tensor.
    Returns the extended batch.
    """
    # batch.shape[0] is always smaller or equal to batch_size
    # if smaller we want to add data to the batch from the
    # dataloader until the batch has the correct size.
    while batch.shape[0] != batch_size:
        # Not the correct size, so we create an iterator
        # that can iterate over the batches in the dataloader
        dataloader_iterator = iter(dataloader)

        # Load the next batch
        nw_batch = next(dataloader_iterator)

        # if the the number of instances in nw_batch + original
        # batch is larger than batch size, then we need to remove
        # the appropriate number of instances from the nw_batch.
        if nw_batch.shape[0] + batch.shape[0] > batch_size:
            # Keep only the first batch_size - batch.shape[0] instances.
            nw_batch = nw_batch[:batch_size - batch.shape[0]]

        # catenate the original batch we the new data. rowbind.
        batch = torch.cat([batch, nw_batch], 0)

    return batch


def extend_batch_tuple(batch, dataloader, batch_size):
    """
    The same as extend_batch, but here the batch is a list of tensors
    to be extended. All tensors are assumed to have the same first dimension.
    Returns the extended batch (i. e. list of extended tensors).
    """
    while batch[0].shape[0] != batch_size:
        dataloader_iterator = iter(dataloader)
        nw_batch = next(dataloader_iterator)
        if nw_batch[0].shape[0] + batch[0].shape[0] > batch_size:
            nw_batch = [nw_t[:batch_size - batch[0].shape[0]]
                        for nw_t in nw_batch]
        batch = [torch.cat([t, nw_t], 0) for t, nw_t in zip(batch, nw_batch)]
    return batch


def get_validation_iwae(val_dataloader, mask_generator, batch_size,
                        model, num_samples, verbose=False):
    """
    Compute mean IWAE log likelihood estimation of the validation set.
    Takes validation dataloader, mask generator, batch size, model (VAEAC)
    and number of IWAE latent samples per object.
    Returns one float - the estimation.

    IWAE is an abrevation for Importance Sampling Estimator
    log p_{theta, psi}(x|y) \approx
    log {1/S * sum_{i=1}^S [p_theta(x|z_i, y) * p_psi(z_i|y) / q_phi(z_i|x,y)]}
    where z_i ~ q_phi(z|x,y)
    """

    # Set variables to store the number of instances evaluated and avg_iwae
    cum_size = 0
    avg_iwae = 0

    # Set the validation dataloader to be the iterator
    iterator = val_dataloader

    # Give the user some feedback by creating a progress bar
    if verbose:
        iterator = tqdm(iterator)

    # Iterate over all the batches in the validation set
    for batch in iterator:

        # Get the number of instances in the current batch
        init_size = batch.shape[0]

        # Extend the batch if init_size not equal to batch_size
        # returns the batch if appropriate size or adds instances
        # from validation dataloader
        batch = extend_batch(batch, val_dataloader, batch_size)

        # Create the mask for the current batch.
        # It is a MCARGenerator(0.2). 20%chance that each
        # individual element[i,j] in the batch is set to be masked.
        # mask consists of zeros (observed) and ones (missing or masked)
        mask = mask_generator(batch)

        # If the model.parameters are located on a Nivida GPU, then
        # we send batch and mask to GPU, as it is faster than CPU.
        if next(model.parameters()).is_cuda:
            batch = batch.cuda()
            mask = mask.cuda()

        # use with torch.no_grad() as we in this with clause do not
        # want to compute the gradients, as they are not important
        # / not needed for doing backpropagation.
        with torch.no_grad():
            # Get the iwae for each instance in the current batch
            # but save only the first init_size, as the other are
            # just arbitrary instances we "padded" the batch with
            # to get the appropriate shape.
            iwae = model.batch_iwae(batch, mask, num_samples)[:init_size]

            # Update the average iwae over all batches (over all instances)
            # This is called recursive/online updating of the mean.
            # I have verified the method. Takes the
            # old average * cum_size to get old sum of iwae
            # adds the sum of newly computed iwae. Then divide the
            # total iwae by the number of instances: cum_size + iwae.shape[0])
            avg_iwae = (avg_iwae * (cum_size / (cum_size + iwae.shape[0])) +
                        iwae.sum() / (cum_size + iwae.shape[0]))

            # Update the number of instances evaluated
            cum_size += iwae.shape[0]

        # Update the description (text before progress bar) in the console
        if verbose:
            iterator.set_description('Validation IWAE: %g' % avg_iwae)

    # return the average iwae over all instances in the validation set.
    return float(avg_iwae)
