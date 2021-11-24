import math

import torch
from torch.distributions import kl_divergence
from torch.nn import Module

from prob_utils import normal_parse_params


class VAEAC(Module):
    """
    Variational Autoencoder with Arbitrary Conditioning core model.
    It is rather flexible, but have several assumptions:
    + The batch of objects and the mask of unobserved features
      have the same shape.
    + The prior and proposal distributions in the latent space
      are component-wise independent Gaussians.
    The constructor takes
    + Prior and proposal network which take as an input the concatenation
      of the batch of objects and the mask of unobserved features
      and return the parameters of Gaussians in the latent space.
      The range of neural networks outputs should not be restricted.
    + Generative network takes latent representation as an input
      and returns the parameters of generative distribution
      p_theta(x_b | z, x_{1 - b}, b), where b is the mask
      of unobserved features. The information about x_{1 - b} and b
      can be transmitted to generative network from prior network
      through nn_utils.MemoryLayer. It is guaranteed that for every batch
      prior network is always executed before generative network.
    + Reconstruction log probability. rec_log_prob is a callable
      which takes (groundtruth, distr_params, mask) as an input
      and return vector of differentiable log probabilities
      p_theta(x_b | z, x_{1 - b}, b) for each object of the batch.
    + Sigma_mu and sigma_sigma are the coefficient of the regularization
      in the hidden space. The default values correspond to a very weak,
      almost disappearing regularization, which is suitable for all
      experimental setups the model was tested on.
    """
    def __init__(self, rec_log_prob, proposal_network, prior_network,
                 generative_network, sigma_mu=1e4, sigma_sigma=1e-4):
        super().__init__()
        self.rec_log_prob = rec_log_prob
        self.proposal_network = proposal_network
        self.prior_network = prior_network
        self.generative_network = generative_network
        self.sigma_mu = sigma_mu
        self.sigma_sigma = sigma_sigma

    def make_observed(self, batch, mask):
        """
        Copy batch of objects and zero unobserved features.
        """
        # Create the observed values by combining the batch and the mask
        # Create a copy/new tensor identical to the batch
        # observed = torch.tensor(batch)
        observed = batch.clone().detach()

        # Apply the mask. Everywhere the mask is one, we set
        # the observed values to zero. Mask is float, so we
        # take .byte() to get the uin8 values (int between 0-255)
        # Takes floor() and then covert to int.
        # observed[mask.byte()] = 0
        observed[mask.byte() == 1] = 0

        return observed

    def make_latent_distributions(self, batch, mask, no_proposal=False):
        """
        Make latent distributions for the given batch and mask.
        No no_proposal is True, return None instead of proposal distribution.
        """
        # Start by creating the observed values where the masked is applied
        # such that the masked values are 0.
        observed = self.make_observed(batch, mask)

        # NOT COMPLETELY SURE WHAT NO_PROPOSAL MEANS
        # it is when we are not interested in the normal distribution
        # of the latent space computed based on the proposal network
        # Used when we just want to generate new samples based only on the prior.
        if no_proposal:
            proposal = None

        else:
            # Default to go in here

            # Create the full_info, which is the concatenated version
            # of the batch and mask. So we colbind them together.
            # Batch and mask have same shape: batch_size x sum(one_hot_max_sizes)
            # Which is batch_size x num_feature for the continuous case.
            # full_info have dimensions  batch_size x 2*sum(one_hot_max_sizes)
            # THIS IS A DEEP COPY
            full_info = torch.cat([batch, mask], 1)

            # Send the full_information through the proposal network
            # the encoder. It needs the full information to know if a
            # value is missing or just masked.
            # We get out a matrix of size: batch_size x 64*2
            # where 64 is the dimension of the latent space.
            # For each dimension we get a mean mu and a sd sigma.
            # the first 64 values are the mus and the last
            # 64 are the softplus of the sigmas, so it can take on any value.
            # softplus(x) = ln(1+e^{x})
            proposal_params = self.proposal_network(full_info)

            # Takes the proposal_parameters and returns a normal distribution,
            # which is component-wise independent.
            # If sigma (after softmax transform) is less than 1e-3,
            # then we set sigma to this value.
            proposal = normal_parse_params(proposal_params, 1e-3)

        # The we compute the normal parameters of the prior network
        # i.e., the third network that we are interested in
        # the one that computes the conditional dist where some
        # of the features are masked.
        # So instead of sending in the batch values/the truth
        # we send in the observed values and the mask.
        prior_params = self.prior_network(torch.cat([observed, mask], 1))

        # Create the normal distribution based on the parameters
        # (mu, sigma) from the prior_network
        prior = normal_parse_params(prior_params, 1e-3)

        # Return the two multivariate normal distributions.
        return proposal, prior

    def prior_regularization(self, prior):
        """
        The prior distribution regularization in the latent space.
        Though it saves prior distribution parameters from going to infinity,
        the model usually doesn't diverge even without this regularization.
        It almost doesn't affect learning process near zero with default
        regularization parameters which are recommended to be used.
        """

        # Get the number of instances in the batch
        num_objects = prior.mean.shape[0]

        # Get the means and makes sure that it has the correct shape
        mu = prior.mean.view(num_objects, -1)

        # Get the sigmas and make sure dim: num_objects x 64
        sigma = prior.scale.view(num_objects, -1)

        # Here we compute the regularization terms:
        # NEED TO LOOK AT THIS MORE!!!
        # The come from the middle of the 5th page
        # Recall that self.sigma_mu = 1e4
        # sum the rows, where the elements are squared.
        mu_regularizer = -(mu ** 2).sum(-1) / 2 / (self.sigma_mu ** 2)
        # self.sigma_sigma = 1e-4
        sigma_regularizer = (sigma.log() - sigma).sum(-1) * self.sigma_sigma

        # Add the regularization terms together
        # and they are later added to the model log-likelihood.
        return mu_regularizer + sigma_regularizer

    def batch_vlb(self, batch, mask):
        """
        Compute differentiable lower bound for the given batch of objects
        and mask.
        """
        # Want to compute the variational lower bound of the given batch
        # which is used for training the networks

        # Compute the normal distributions obtained from
        # the proposal and prior networks
        proposal, prior = self.make_latent_distributions(batch, mask)

        # Apply the regularization on the mus and sigmas of
        # normal dist obtained from the prior networks such that
        # they dont blow up. Regularized according to their
        # normalgamma prior.
        prior_regularization = self.prior_regularization(prior)

        # WE NEED TO USE RSAMPLE() AND NOT SAMPLE()
        # As we use the reparameterization trick, which allows
        # backpropagation through the mean and standard deviation layers.
        # https://pytorch.org/docs/stable/distributions.html#pathwise-derivative
        # For each instance in the batch we sample 64 values, for the
        # 64 dimensions in the latent space.
        # So we sample one sample for each instance
        latent = proposal.rsample()

        # Send these latent coordinates through the generative network
        # and get the batch_size x 2*num_features. (in cont case)
        # where we for each row have a normal dist on each feature
        # The form will be (mu_1, sigma_1, ..., mu_12, sigma_12)
        rec_params = self.generative_network(latent)

        # Compute the reconstruction loss, i.e., the log
        # likelihood of only the masked values in the batch (true values)
        # given the current reconstruction parameters from the
        # generative network/decoder. We do not consider the
        # log lik of observed or missing/nan values.
        rec_loss = self.rec_log_prob(batch, rec_params, mask)

        # Compute the KL divergence between the two normal distributions
        # obtained from the proposal (encoder) network and
        # the prior (cond dist) network.
        # Since we are assuming full-factorized Gaussian proposal dist,
        # we can compute KL analytically.
        # Computes KL(p, q) = \int p(x) log(p(x)/q(x)) dx
        # = 0.5 * { (sigma_p/sigma_q)^2 + (mu_q - mu_p)^2/sigma_q^2 - 1 + 2 ln (sigma_q/sigma_p)}
        # when both p and q are normal.
        # The output is batch_size x 64 (in cont case) where each row i are
        # the 64 KL div for each of the 64 dim in the latent space for instance i.
        # Then we sum over the rows, so that we add the KL for each dim together
        # to get the total kl div for each of the instances.
        kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).sum(-1)

        # Obtain the variational lower bound as as given in eq 6 on page 5.
        # We want to maximize this lower bound.
        # This is a tensor of length batch_size.
        return rec_loss - kl + prior_regularization

    def batch_iwae(self, batch, mask, K):
        """
        Compute IWAE log likelihood estimate with K samples per object.
        Technically, it is differentiable, but it is recommended to use it
        for evaluation purposes inside torch.no_grad in order to save memory.
        With torch.no_grad the method almost doesn't require extra memory
        for very large K.
        The method makes K independent passes through generator network,
        so the batch size is the same as for training with batch_vlb.


        IWAE is an abrevation for Importance Sampling Estimator
        log p_{theta, psi}(x|y) \approx
        log {1/K * sum_{i=1}^K [p_theta(x|z_i, y) * p_psi(z_i|y) / q_phi(z_i|x,y)]} =
        log {sum_{i=1}^K exp(log[p_theta(x|z_i, y) * p_psi(z_i|y) / q_phi(z_i|x,y)])} - log(K) =
        log {sum_{i=1}^K exp(log[p_theta(x|z_i, y)] + log[p_psi(z_i|y)] - log[q_phi(z_i|x,y)])} - log(K) =
        logsumexp(log[p_theta(x|z_i, y)] + log[p_psi(z_i|y)] - log[q_phi(z_i|x,y)]) - log(K) =
        logsumexp(rec_loss + prior_log_prob - proposal_log_prob) - log(K),
        where z_i ~ q_phi(z|x,y)
        """
        # Get two normal distributions of dimension 64, where the
        # parameters are obtained from the proposal and prior networks.
        proposal, prior = self.make_latent_distributions(batch, mask)
        estimates = []
        for i in range(K):
            # Create samples from the proposal network (the encoder).
            # I.e., z_i ~ q_phi(z|x,y)
            latent = proposal.rsample()  # See equation 18 on page 18.

            # Then we compute/decode the latent variables by sending the
            # means and the sigmas through the generative network.
            # We end up with parameters and NOT FINAL INSTANCES!
            # These are the reconstruction parameters.
            rec_params = self.generative_network(latent)

            # Compute the reconstruction loss.
            # It computes the log-likelihood of observing
            # the truth (batch) given the parameters in rec_params.
            # The loss consist only of the log-lik of masked values.
            rec_loss = self.rec_log_prob(batch, rec_params, mask)

            # Log_prob compute the log-likelihood of the normal dist given the latent data.
            # I.e., the log of the probability density/mass function evaluated at latent.
            # Where latent are random samples sampled from the encoder.
            prior_log_prob = prior.log_prob(latent)  # This is 64x64
            # This makes sure that the dimensions are batch.shape[0] x something
            # no effect for full batches at least
            prior_log_prob = prior_log_prob.view(batch.shape[0], -1)
            # n - 1 = 2 - 1 = 1. So could have used sum(1).
            # Sum over the rows, i.e., add the log-likelihood for each instance
            prior_log_prob = prior_log_prob.sum(-1)

            # Same here as above.
            proposal_log_prob = proposal.log_prob(latent)
            proposal_log_prob = proposal_log_prob.view(batch.shape[0], -1)
            proposal_log_prob = proposal_log_prob.sum(-1)

            # Combine these results to obtain the estimated loss.
            # This formula arises from equation 18 on page 18.
            # Consists of batch.shape[0] number of values
            #
            estimate = rec_loss + prior_log_prob - proposal_log_prob

            # Convert it to a batch.shape[0] x 1 matrix,
            # and add it to the estimates.
            estimates.append(estimate[:, None])

        # Concatenate the estimates so we get a tensor of dim
        # batch.shape[0] x K. Then we take the log sum exp
        # along the rows (1 axis, as 0 axis is downwards),
        # then we subtract - log(K). Guess this is a trick,
        # but unsure. Stabalizing trick.
        # Actually divide by K since we want average for each
        # instance, but with log this turns into -log(K).
        return torch.logsumexp(torch.cat(estimates, 1), 1) - math.log(K)

    def generate_samples_params(self, batch, mask, K=1):
        """
        Generate parameters of generative distributions for samples
        from the given batch.
        It makes K latent representation for each object from the batch
        and generate samples from them.
        The second axis is used to index samples for an object, i. e.
        if the batch shape is [n x D1 x D2], then the result shape is
        [n x K x D1 x D2].
        It is better to use it inside torch.no_grad in order to save memory.
        With torch.no_grad the method doesn't require extra memory
        except the memory for the result.
        """

        # Start by creating the latent representation obtained from the
        # prior network. The proposal/encoder, we do not care about,
        # hence we set that to "_", which is common for variables that
        # will not be used.
        # prior now contains 64 mus and 64 sigmas for each observation
        # in the batch. If latent dim is 64. This do not distinguish
        # between if features are categorical or numerical.
        # The mask her only contains missing values which we want to impute.
        _, prior = self.make_latent_distributions(batch, mask)

        # Create a list to keep the sampled parameters
        samples_params = []

        # Iterate over the number of imputations we want to make for each instance
        for i in range(K):

            # Sample one latent representation for each instance in the batch.
            latent = prior.rsample()

            # Then we send the latent representation through the generative
            # network and obtain the sample_params which are mu and sigma
            # for each of the features of the original data set. (in each row)
            # One the form (mu_1, sigma_1, ... mu_n, sigma_n)
            # A bit different for the categorical data as we then get
            # prob_i_0, prob_i_1, ..., prob_i_{K-1} if ith feature have K classes.
            sample_params = self.generative_network(latent)

            # unsqueeze(1) increase the dimension of sample_params, such that
            # (batch_size, 2*num_features) becomes (batch_size, 1, 2*num_features)
            # Then add this tensor to the list
            samples_params.append(sample_params.unsqueeze(1))

        # Concatenate the list in the first dimension, so that we get a tensor
        # of dimensions (batch_size, K, 2*num_features)
        return torch.cat(samples_params, 1)

    def generate_reconstructions_params(self, batch, mask, K=1):
        """
        Generate parameters of generative distributions for reconstructions
        from the given batch.
        It makes K latent representation for each object from the batch
        and generate samples from them.
        The second axis is used to index samples for an object, i. e.
        if the batch shape is [n x D1 x D2], then the result shape is
        [n x K x D1 x D2].
        It is better to use it inside torch.no_grad in order to save memory.
        With torch.no_grad the method doesn't require extra memory
        except the memory for the result.
        """
        _, prior = self.make_latent_distributions(batch, mask)
        reconstructions_params = []
        for i in range(K):
            latent = prior.rsample()
            rec_params = self.generative_network(latent)
            reconstructions_params.append(rec_params.unsqueeze(1))
        return torch.cat(reconstructions_params, 1)
