import torch
from torch.distributions import Categorical, Normal
from torch.nn import Module
from torch.nn.functional import softplus, softmax


def normal_parse_params(params, min_sigma=0):
    """
    Take a Tensor (e. g. neural network output) and return
    torch.distributions.Normal distribution.
    This Normal distribution is component-wise independent,
    and its dimensionality depends on the input shape.
    First half of channels is mean of the distribution,
    the softplus of the second half is std (sigma), so there is
    no restrictions on the input tensor.

    min_sigma is the minimal value of sigma. I. e. if the above
    softplus is less than min_sigma, then sigma is clipped
    from below with value min_sigma. This regularization
    is required for the numerical stability and may be considered
    as a neural network architecture choice without any change
    to the probabilistic model.
    """
    # Get the number of instances
    n = params.shape[0]
    # Then get the dimension of the parameters
    d = params.shape[1]
    # Use double dash to get integer. Do not need it as we by construction always have 2*num_dim_latent_space
    mu = params[:, :d // 2]  # Get the first halves which are the means
    sigma_params = params[:, d // 2:]  # Get the second half which are transformed sigmas
    sigma = softplus(sigma_params)  # ln(1 + exp(sigma_params))
    sigma = sigma.clamp(min=min_sigma)  # Make sure that sigma >= min_sigma
    # Create the normal dist. Multivariate, but with independent dimensions. Correlation = 0.
    distr = Normal(mu, sigma)
    return distr


def categorical_parse_params_column(params, min_prob=0):
    """
    Take a Tensor (e. g. a part of neural network output) and return
    torch.distributions.Categorical distribution.
    The input tensor after applying softmax over the last axis contains
    a batch of the categorical probabilities. So there are no restrictions
    on the input tensor.

    Technically, this function treats the last axis as the categorical
    probabilities, but Categorical takes only 2D input where
    the first axis is the batch axis and the second one corresponds
    to the probabilities, so practically the function requires 2D input
    with the batch of probabilities for one categorical feature.

    min_prob is the minimal probability for each class.
    After clipping the probabilities from below they are renormalized
    in order to be a valid distribution. This regularization
    is required for the numerical stability and may be considered
    as a neural network architecture choice without any change
    to the probabilistic model.
    """
    # Apply the softmax: Softmax(x_i) = (exp(x_i))/(\sum_{j} exp(x_j))
    # Here x_i are the logits and can take on any value, negative and positive.
    # The output Softmax(x_i) \in [0,1] and \sum_{j} Softmax(x_i) = 1.
    params = softmax(params, -1)

    # If the probability/param is smaller than min_prob, we set it to zero.
    params = params.clamp(min_prob)

    # Make sure that parms sum to 1 after the clamping.
    params = params / params.sum(-1, keepdim=True)

    # Then create a categorical distribution which will have len(params)
    # number of categories and the probability for each of them is given in params.
    distr = Categorical(probs=params)
    return distr


class GaussianLoss(Module):
    """
    Compute reconstruction log probability of groundtruth given
    a tensor of Gaussian distribution parameters and a mask.
    Gaussian distribution parameters are output of a neural network
    without any restrictions, the minimal sigma value is clipped
    from below to min_sigma (default: 1e-2) in order not to overfit
    network on some exact pixels.

    The first half of channels corresponds to mean, the second half
    corresponds to std. See normal_parse_parameters for more info.
    This layer doesn't work with NaNs in the data, it is used for
    inpainting. Roughly speaking, this loss is similar to L2 loss.
    Returns a vector of log probabilities for each object of the batch.
    """
    def __init__(self, min_sigma=1e-2):
        super().__init__()
        self.min_sigma = min_sigma

    def forward(self, groundtruth, distr_params, mask):
        # Create the normal dist. Multivariate, but with independent dimensions. Correlation = 0.
        # Note that the sigmas in the distr_params are sent through the softplus function
        # such that the values always are positive.
        distr = normal_parse_params(distr_params, self.min_sigma)

        # The the log-likelihoods of observing the truth given the estimated distributions.
        # Multiply with mask, as we only want to evaluate the quality of the missing values
        # which we want to impute.
        log_probs = distr.log_prob(groundtruth) * mask

        # Sum the log-likelihoods.
        return log_probs.view(groundtruth.shape[0], -1).sum(-1)


class GaussianCategoricalLoss(Module):
    """
    This layer computes log probability of groundtruth for each object
    given the mask and the distribution parameters.
    This layer works for the cases when the dataset contains both
    real-valued and categorical features.

    one_hot_max_sizes[i] is the one-hot max size of i-th feature,
    if i-th feature is categorical, and 0 or 1 if i-th feature is real-valued.
    In the first case the distribution over feature is categorical,
    in the second case it is Gaussian.

    For example, if one_hot_max_sizes is [4, 1, 1, 2], then the distribution
    parameters for one object is the vector
    [p_00, p_01, p_02, p_03, mu_1, sigma_1, mu_2, sigma_2, p_30, p_31],
    where Softmax([p_00, p_01, p_02, p_03]) and Softmax([p_30, p_31])
    are probabilities of the first and the fourth feature categories
    respectively in the model generative distribution, and
    Gaussian(mu_1, sigma_1 ^ 2) and Gaussian(mu_2, sigma_2 ^ 2) are
    the model generative distributions on the second and the third features.

    For the definitions of min_sigma and min_prob see normal_parse_params
    and categorical_parse_params docs.

    This layer works correctly with missing values in groundtruth
    which are represented by NaNs.

    This layer works with 2D inputs only.
    """
    def __init__(self, one_hot_max_sizes, min_sigma=1e-4, min_prob=1e-4):
        super().__init__()
        self.one_hot_max_sizes = one_hot_max_sizes
        self.min_sigma = min_sigma
        self.min_prob = min_prob

    def forward(self, groundtruth, distr_params, mask):
        # Which column in distr_params we now consider.
        # Either increases with 2 in cont case (mu and sigma)
        # or in increases with one-hot encoding size in cat case.
        cur_distr_col = 0

        # List to store the log probabilities.
        log_prob = []
        for i, size in enumerate(self.one_hot_max_sizes):
            # for i-th feature
            if size <= 1:
                # Gaussian distribution

                # select groundtruth, mask and distr_params for i-th feature
                groundtruth_col = groundtruth[:, i: i + 1]  # Look at the ith column of the truth
                mask_col = mask[:, i: i + 1]  # Get the ith column of the mask
                # these are the mean and sigma for the ith feature,
                # so dimensions batch_size x 2
                params = distr_params[:, cur_distr_col: cur_distr_col + 2]
                cur_distr_col += 2

                # generative model distribution for the feature
                distr = normal_parse_params(params, self.min_sigma)

                # copy groundtruth column, so that zeroing nans will not
                # affect the original data
                # gt_col_nansafe = torch.tensor(groundtruth_col)
                gt_col_nansafe = groundtruth_col.clone().detach()
                # If groundtruth dont have any nans then this line does not change anything
                nan_mask = torch.isnan(groundtruth_col)
                gt_col_nansafe[nan_mask] = 0
                # Everythin that was nan is now 0.

                # Mask_col masks both the nan/missing values
                # and the artificially masked values.
                # We want to compute the the log prob
                # only over the artificially missing
                # features, so we omit the missing values.
                # Bottom og page 5.

                # So we remove the masking of the missing values
                # So those ones in mask_col which are there due
                # to missing values are now turned in to zeros.
                mask_col = mask_col * (torch.logical_not(nan_mask)).float()


                # Get the log-likelihood, but only of the masked values
                # i.e., the ones hat are masked by the masking filter MCARGenerator
                # So this one is batch_size x 1.
                # So this is the log-lik of observing the ground truth
                # given the current parameters, for only the
                # artificially masked features.
                col_log_prob = distr.log_prob(gt_col_nansafe) * mask_col
            else:
                # categorical distribution

                # select groundtruth, mask and distr_params for i-th feature
                groundtruth_col = groundtruth[:, i]
                mask_col = mask[:, i]
                params = distr_params[:, cur_distr_col: cur_distr_col + size]
                cur_distr_col += size

                # generative model distribution for the feature
                # distr is a "torch.distributions.Categorical distribution"
                distr = categorical_parse_params_column(params, self.min_prob)

                # copy groundtruth column, so that zeroing nans will not
                # affect the original data
                # gt_col_nansafe = torch.tensor(groundtruth_col)
                gt_col_nansafe = groundtruth_col.clone().detach()
                nan_mask = torch.isnan(groundtruth_col)
                gt_col_nansafe[nan_mask] = 0

                # compute the mask of the values
                # which we consider in the log probability
                mask_col = mask_col * (torch.logical_not(nan_mask)).float()

                col_log_prob = distr.log_prob(gt_col_nansafe) * mask_col
                col_log_prob = col_log_prob[:, None]

            # append the column of log probabilities for the i-th feature
            # (for those instances that are masked) into log_prob list
            log_prob.append(col_log_prob)
            # log_prob is now a list of length num_features, where each
            # element is a tensor batch_size x 1 containing the log-lik
            # of the parameters of masked values.

        # concatenate the list so we get a tensor of dim batch x features
        # Then we sum along the the rows. i.e., for each observation in the
        # bach. So a tensor of length batch size.
        return torch.cat(log_prob, 1).sum(-1)


class CategoricalToOneHotLayer(Module):
    """
    This layer expands categorical features into one-hot vectors, because
    multi-layer perceptrons are known to work better with this data
    representation. It also replaces NaNs with zeros in order so that
    further layers may work correctly.

    one_hot_max_sizes[i] is the one-hot max size of i-th feature,
    if i-th feature is categorical, and 0 or 1 if i-th feature is real-valued.

    add_nan_maps_for_columns is an optional list which contains
    indices of columns which isnan masks are to be appended
    to the result tensor. This option is necessary for proposal
    network to distinguish whether value is to be reconstructed or not.
    """
    def __init__(self, one_hot_max_sizes, add_nans_map_for_columns=[]):
        super().__init__()
        # Here one_hot_max_sizes includes zeros at the end of the list
        # one_hot_max_sizes + [0] * len(one_hot_max_sizes)
        # So if we have that featuers have this many categories [1, 2, 3, 1],
        # then we get that one_hot_max_sizes = [1, 2, 3, 1, 0, 0, 0, 0]
        self.one_hot_max_sizes = one_hot_max_sizes

        # Is always an empty column for the prior network
        # while it is a list [0, 1, ..., length(one_hot_max_sizes)-1)
        # for the proposal network.
        # So for the proposal network we apply the nan masks to each column/feature
        self.add_nans_map_for_columns = add_nans_map_for_columns

    def forward(self, input):
        # Input is torch.cat([batch, mask], 1), so a matrix of
        # dimension batch_size x 2*sum(one_hot_max_sizes)
        # At least for continuous data where one_hot_max_sizes
        # only consists of ones.
        # ONE_HOT_MAX_SIZES ARE PADDED WITH ZEROS AT THE END!

        # variable to store the outcolumns.
        # that is the input columns / one hot encoding
        # + is nan.mask.
        out_cols = []

        # We iterate over the features and get the number
        # of categories for each feature.
        # so i goes from 0 to 2*num_features-1
        # For i in [num_features, 2*num_features-1] will have size <= 1,
        # even for categorical features.
        for i, size in enumerate(self.one_hot_max_sizes):

            # If size <= 1, then the feature is continuous.
            if size <= 1:
                # real-valued feature
                # just copy it and replace NaNs with zeros
                # OR, the last half of self.one_hot_max_sizes
                #

                # Take the ith column of the input
                # NOTE THAT THIS IS NOT A DEEP COPY
                out_col = input[:, i: i + 1]
                # use i:i+1 to get a column vector, instead of just i
                # which gives a row vector

                # check if any of the values are nan, i.e., missing
                nan_mask = torch.isnan(out_col)

                # set all the missing values to 0.
                # SO THIS CHANGES THE INPUT VARIABLE: the full_info matrix
                out_col[nan_mask] = 0
            else:
                # categorical feature
                # replace NaNs with zeros

                # Get the categories for each instance for the ith feature
                # start to count at zero. So if we have 2 cat, then this
                # vector will contains zeros and ones.
                """
                /Users/larsolsen/Desktop/PhD/VAEAC_Updated_Version/prob_utils.py:312:
                UserWarning: To copy construct from a tensor, it is recommended to use
                sourceTensor.clone().detach() or
                sourceTensor.clone().detach().requires_grad_(True),
                rather than torch.tensor(sourceTensor).
                cat_idx = torch.tensor(input[:, i])
                """
                cat_idx = torch.tensor(input[:, i])
                # Line below does not work. Use the one above even thoug it gives a warning.
                # cat_idx = input[:, i].clone().detach().requires_grad_(True)
                # Check if any of the categories are nan / missing
                nan_mask = torch.isnan(cat_idx)
                # Set the nan values to 0
                cat_idx[nan_mask] = 0

                # # one-hot encoding
                # Get the number of instances
                n = input.shape[0]
                # create a matrix, where the jth row is the one-hot
                # encoding of the ith feature of the jth instance.
                out_col = torch.zeros(n, size, device=input.device)

                # Include long() to get it to int64 format.  which it is by default
                # SO THIS IS NOT NEEDED!
                out_col[torch.arange(n).long(), cat_idx.long()] = 1
                # So if category is 2 (1 in cat_idx) out of 4. then we get [0, 1, 0, 0]
                # RECALL that the categories go from 0 to K-1.

                # set NaNs to be zero vectors
                out_col[nan_mask] = 0

                # reshape nan_mask to be a column
                nan_mask = nan_mask[:, None]

            # append this feature column to the result
            # out_col is n x size =
            # batch_size x num_categories_for_this_feature
            out_cols.append(out_col)

            # if necessary, append isnan mask of this feature to the result
            # which we always do for the proposal network.
            # This only happens for the first half of the i's,
            # so for i = 0, 1, ..., num_features - 1.
            if i in self.add_nans_map_for_columns:
                # so we add the columns of nan_mask
                out_cols.append(nan_mask.float())

        # out_cols now is a list of num_features tensors of shape n x size
        # = n x 1 for continuous variables. So we concatenate them
        # to get a matrix of dim n x 2*num_features (in cont case) for
        # prior net, but for proposal net, it is n x 3*num_features
        # They take the form  [batch1, is.nan1, batch2, is.nan2, …,
        # batch12, is.nan12, mask1, mask2, …, mask12]
        return torch.cat(out_cols, 1)


class GaussianCategoricalSampler(Module):
    """
    Generates a sample from the generative distribution defined by
    the output of the neural network.

    one_hot_max_sizes[i] is the one-hot max size of i-th feature,
    if i-th feature is categorical, and 0 or 1 if i-th feature is real-valued.

    The distribution parameters format, min_sigma and min_prob are described
    in docs for GaussianCategoricalLoss.

    If sample_most_probable is True, then the layer will return
    mean for Gaussians and the most probable class for categorical features.
    Otherwise the fair sampling procedure from Gaussian and categorical
    distributions is performed.
    """
    def __init__(self, one_hot_max_sizes, sample_most_probable=False,
                 min_sigma=1e-4, min_prob=1e-4):
        super().__init__()

        # one-hot max size of i-th feature, if i-th feature is categorical,
        # and 0 or 1 if i-th feature is real-valued.
        self.one_hot_max_sizes = one_hot_max_sizes

        # We set this to TRUE when we use this class.
        # So just return the mean for Gaussian.
        # This is not what we want in case of SHAPLEY.
        # SO WE SHOULD USE FALSE
        self.sample_most_probable = sample_most_probable

        # We use the default values, both 1e-4.
        self.min_sigma = min_sigma
        self.min_prob = min_prob

    def forward(self, distr_params):
        # dist_params is a matrix of form batch_size x (mu_1, sigma_1, ..., mu_n, sigma_n)
        # A bit different for categorical features as we
        # there do not have mu and sigma for generative normal at the end,
        # but rather logits for the categorical distribution.

        # A counter to keep track of which
        cur_distr_col = 0

        # List to store all the samples sampled from the
        # normal distribution with parameters from distr_params.
        sample = []

        # iterate over the number of features and the number
        # of categories for each feature.
        for i, size in enumerate(self.one_hot_max_sizes):

            # If we are dealing with continuous features
            if size <= 1:
                # Gaussian distribution
                # Get the mu and sigma for the current feature, for each instance
                params = distr_params[:, cur_distr_col: cur_distr_col + 2]
                cur_distr_col += 2

                # generative model distribution for the feature
                # so create batch_size number of normal distributions.
                distr = normal_parse_params(params, self.min_sigma)

                # If we are going to sample the mean or do a random sampling
                if self.sample_most_probable:
                    col_sample = distr.mean
                else:
                    col_sample = distr.sample()
            else:
                # Categorical distribution

                # Extract the logits for the different categories
                # of the ith categorical variable
                params = distr_params[:, cur_distr_col: cur_distr_col + size]
                cur_distr_col += size

                # Generative model distribution for the feature
                # So convert from logits to probabilites.
                # distr is a "torch.distributions.Categorical distribution"
                distr = categorical_parse_params_column(params, self.min_prob)

                # If we are going to use the most likely category,
                # or if we are to sample based on the probabilities
                # for each of the classes.
                if self.sample_most_probable:
                    col_sample = torch.max(distr.probs, 1)[1][:, None].float()
                else:
                    col_sample = distr.sample()[:, None].float()

            # Add the vector of sampled values for the i´th
            # feature to the sample list.
            sample.append(col_sample)

        # Create a matrix by column binding the vectors in the list
        return torch.cat(sample, 1)


class SetGaussianSigmasToOne(Module):
    """
    This layer is used in missing features imputation. Because the target
    metric for this problem is NRMSE, we set all sigma to one, so that
    the optimized metric is L2 loss without any disbalance between features,
    which probably increases (we are not sure about this) NRMSE score.

    one_hot_max_sizes[i] is the one-hot max size of i-th feature,
    if i-th feature is categorical, and 0 or 1 if i-th feature is real-valued.

    The distribution parameters format is described in docs
    for GaussianCategoricalLoss.

    Because the output of the network is passed through softplus
    for real-valued features, this layer replaces all corresponding
    columns with softplus^{-1}(1), where softplus^{-1} is the inverse
    softplus function.
    """
    def __init__(self, one_hot_max_sizes):
        super().__init__()
        # This one_hot_max_sizes are not padded with zeros at the end.
        self.one_hot_max_sizes = one_hot_max_sizes

    def forward(self, distr_params):
        # Variable to track the columns for the current feature.
        cur_distr_col = 0

        # mask shows which columns must save their values
        mask = torch.ones(distr_params.shape[1], device=distr_params.device)

        # Iterate over the features and get the number of categories for each feature.
        for i, size in enumerate(self.one_hot_max_sizes):
            if size <= 1:
                # Continuous setting, so each feature have two parameters, mu and sigma
                # set every other value to zero / the sigmas
                mask[cur_distr_col + 1] = 0

                # Increase by two
                cur_distr_col += 2
            else:
                # Here we are now changing the sigma, since we are dealing with
                # size number of classes.
                cur_distr_col += size

        # log(exp(1)-1) = 0.54125
        # Create vector of log(exp(1)-1) with same size as distr_params
        inverse_softplus = torch.ones_like(distr_params) * 0.54125

        # Use [None] to convert from 1 dim vector of length 2*num_features
        # to a matrix of dim 1 X (2*num_features)
        # Insert log(exp(1)-1) everywhere there is a sigma
        return distr_params * mask[None] + inverse_softplus * (1 - mask)[None]
