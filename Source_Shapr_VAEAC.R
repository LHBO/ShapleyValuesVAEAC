### Load the reticulate library to get python code
library(reticulate)

# Include the shapr library
library(shapr)

# Library needed for doing data manipulations
library(abind)

# User need to specify the path of the folder where
# the user downloaded the GitHub Repository.
# In RStudio one can use 'dirname(rstudioapi::getSourceEditorContext()$path)'.
directory_code = "USER_MUST_SET_PATH_TO_DIRECTORY_WHERE_GitHub_REPOSITORY_IS_STORED"

# The path to where the python build is saved.
# Often something along "/usr/local/bin/python3.6.4",
# depending on operating system.
directory_python = "USER_MUST_SET_PATH_TO_DIRECTORY_WHERE_PYTHON_IS_STORED"

###############################################################
##### INITIATE CONNECTION TO PYTHON BY RETICULATE LIBRARY #####
###############################################################
# Set which version of python to use. We need Python 3.6.4.
# Some later versions work too.
# The needed python libraries must already be installed.
# Use 'py_discover_config()' to see the available versions of 
# Python on you unit.
use_python(directory_python)  

# Since we do not work with full paths, we need to set the
# working directory for the Python program
os = import("os")
os$chdir(directory_code)

# Load the Python functions from the Python file 'ComputeImputations.py'.
source_python(paste(directory_code, 'ComputeImputations.py', sep = '/'))


####################################################################
##### Functions that are needed for fitting the VAEAC approach #####
##### into the shapr-framework.                                #####
####################################################################

# The way to use this code is by three line of code:
# 0. Create data_train, data_test, and fit a model which we want to explain.
# 1. explainer = shapr(data_train, model)
#    This creates the explainer object that has preprossesed the data
#    and created such things as the feature list (coalitions) and much more.
# 2. explainer_added_vaeac = add_vaeac_to_explainer(explainer)
#    This function either adds a path to a fitted vaeac to the 
#    explainer object (if we have trained a vaeac already), 
#    or it fits a vaeac model to the training data.
#    The function takes in parameters for how to train the network.
#    and either saves the model at some specified folder/path, or
#    just uses 'getwd()'.
# 3. explanation_vaeac = explain.vaeac(data_test,
#                                      approach = "vaeac",
#                                      explainer = explainer_added_vaeac,
#                                      prediction_zero = phi_0) (phi_0 = mean(y_train))
#   This computes the shapley values for the test observations in data_test.


# Function that takes in the explainer object and train a VAEAC-model
# based on the parameters provided and then saves the model to disk
# and add the path to the explainer object. If not a path is provided,
# then this path is simply added to the explainer and no training is
# conducted.
# If trained, then records three different VAEACs. 
# The one with the best validation
# error, the one with best average validation error over the last 
# 'running_avg_num_values' epochs and the model after the last epoch.
add_vaeac_to_explainer = function(explainer, 
                                  vaeac_path = NULL,
                                  distribution = "unknown_dist",
                                  param_now = "unknown",
                                  path_to_save_model = NULL,
                                  use_cuda = FALSE,
                                  epochs = 100L,
                                  num_different_vaeac_initiate = 5L,
                                  epochs_initiation_phase = 2L,
                                  masking_ratio = 0.5,
                                  validation_ratio = 0.25,
                                  validation_iwae_num_samples = 40L,
                                  width = 32L,
                                  depth = 3L,
                                  latent_dim = 8L,
                                  lr = 0.001,
                                  batch_size = 64L,
                                  running_avg_num_values = 5L,
                                  verbose = FALSE,
                                  verbose_init = FALSE,
                                  verbose_summary = FALSE,
                                  Annabelle_Simulation_study_code = FALSE,
                                  num_epochs_run_vaeac = NULL) {
  
  # Parameter 'num_epochs_run_vaeac' has been deprecated due to illogical name.
  # replaced by 'epochs_initiation_phase' in new versions from 15.11.21.
  # This is just to make sure that all code that relies on
  # 'num_epochs_run_vaeac' still work.
  if (!is.null(num_epochs_run_vaeac) & is.numeric(num_epochs_run_vaeac)) {
    epochs_initiation_phase = as.integer(num_epochs_run_vaeac)
  }
  
  # If we are provided with a path to a VAEAC model, we can simply attach
  # that to the explainer object. Otherwise we fit a VAEAC model
  if (!is.null(vaeac_path)) {
    # Boolean to explain if the VAEAC model was provided by user
    # 'FALSE' means that we fitted the VAEAC model. 
    explainer$user_provided_VAEAC = TRUE
    explainer$VAEAC$models$best = vaeac_path
  } 
  else {
    # Boolean to explain if the VAEAC model was provided by user
    # 'FALSE' means that we fitted the VAEAC model. 
    explainer$user_provided_VAEAC = FALSE
    
    # Make sure that the relevant are of class integer, 
    # otherwise reticulate and python will not work as expected.
    width                        = as.integer(width)
    depth                        = as.integer(depth)
    epochs                       = as.integer(epochs)
    latent_dim                   = as.integer(latent_dim)
    batch_size                   = as.integer(batch_size)
    epochs_initiation_phase      = as.integer(epochs_initiation_phase)
    running_avg_num_values       = as.integer(running_avg_num_values)
    validation_iwae_num_samples  = as.integer(validation_iwae_num_samples)
    num_different_vaeac_initiate = as.integer(num_different_vaeac_initiate)
    
    # Get the training data
    # HERE WE NEED TO DO SOME EXTRA AS VAEAC CANNOT HANDLE A DATAFRAM
    # AND IT WANT THE CATEGORIES TO BE 0, 1, ..., K-1.
    # SO IT CANNOT HANDLE CATEGORY NAMES
    # NEED TO STORE THAT SOME PLACE
    # SO WE START BY CHECKING IF TRAINING SET CONTAINS CATEGORICAL OR NOT.
    # categorical_col = sapply(seq(ncol(explainer$x_train)), 
    #                                  function(j) is.factor(explainer$x_train[[j]]))
    # categorical_in_data_set = sum(categorical_col) > 0
    categorical_col = unname(explainer$feature_list$classes == "factor")
    categorical_in_data_set = sum(categorical_col) > 0
    
    if (categorical_in_data_set) {
      # Data table contains categorical variables.
      # We need to make certain that the levels start with 0 and increase
      # by one at the time. Otherwise, VAEAC does not work. 
      # So here we check if that assumption holds, if not, we 'correct' 
      # the data into having the right format.
      
      # Convert factors to numeric
      data_train = apply(explainer$x_train, 2, as.numeric)
      
      # Get the indices of the coloumns that are categorical
      categorical_col_ind = seq(length(categorical_col))[categorical_col]
      
      # Iterate over the categorical columns
      for (idx in categorical_col_ind) {
        # Get the current categorical column
        col_now = data_train[,idx] 
        
        # Get the levels of the current categorical feature
        #levels_data = sort(unique(as.character(col_now)))
        levels_data = explainer$feature_list$factor_levels[[idx]]
        
        # Create the new levels starting at 0.
        new_levels = seq(0, length(unique(col_now))-1)
        
        # Insert the updated labels
        data_train[,idx] = as.integer(paste(factor(col_now, labels = new_levels)))
        
        # Need to save the updated levels such that we can invert this transformation
        # later, before we return the shapley values to the user.
        # So keep track on that e.g. "small" -> 0, "large" -> 1, "medium" -> 2
        explainer$feature_list$factor_levels_updated[[explainer$feature_list$labels[idx]]] =
          new_levels
      }
    } else {
      # Data table contains only numerical variables
      data_train = as.matrix(explainer$x_train)
    }
    
    ## Take a look at the new data and see that factors are gone
    # str(explainer$x_train) 
    # str(data_train)
    # table(data_train[,1])
    
    # # Can compare the old and new levels
    # explainer$feature_list$factor_levels
    # explainer$feature_list$factor_levels_updated
    explainer$x_train_updated = data_train

    # Get the one hot max sizes. 
    # Is an array of ones of length p if only continuous variables
    # If a categorical feature have k classes, then its one hot max size is k.
    # In Annabelle's ctree simulation code we might need to add something extra
    if (Annabelle_Simulation_study_code) {
      one_hot_max_sizes = unname(sapply(explainer$model$xlevels, length))
      one_hot_max_sizes[one_hot_max_sizes == 0] = 1
      one_hot_max_sizes = as.integer(one_hot_max_sizes)
    } else {
      one_hot_max_sizes = unname(sapply(explainer$feature_list$factor_levels, length))
      one_hot_max_sizes[one_hot_max_sizes == 0] = 1
      one_hot_max_sizes = as.integer(one_hot_max_sizes)
    }
    
    # Set where to save the vaeac model,
    # if not provided, then set it to the working directory.
    if (is.null(path_to_save_model)) {
      path_to_save_model = getwd()
    }
    
    # Fit/train the VAEAC model. 
    # This is a python function called using Reticulate package.
    fit = train_VAEAC_model(data_train = data_train,
                            distribution = distribution,
                            param_now = param_now,
                            path_to_save_model = path_to_save_model,
                            one_hot_max_sizes = one_hot_max_sizes,
                            use_cuda = use_cuda,
                            epochs = epochs,
                            num_different_vaeac_initiate = num_different_vaeac_initiate,
                            epochs_initiation_phase = epochs_initiation_phase,
                            masking_ratio = masking_ratio,
                            validation_ratio = validation_ratio,
                            validation_iwae_num_samples = validation_iwae_num_samples,
                            width = width,
                            depth = depth,
                            latent_dim = latent_dim,
                            lr = lr,
                            batch_size = batch_size,
                            running_avg_num_values = running_avg_num_values,
                            verbose = verbose,
                            verbose_init = verbose_init,
                            verbose_summary = verbose_summary)
    
    # Extract the trained vaeac models
    vaeac_models_aux = list(best = fit[[1]],
                            running_best = fit[[2]],
                            last = fit[[3]])
    
    # Extract the training/validation results for the VAEAC
    vaeac_models_results_aux = list(train_vlb = fit[[4]],
                                    validation_iwae = fit[[5]],
                                    validation_iwae_running = fit[[6]])
    
    # Create a list of all the parameters used to train the VAEAC-model
    vaeac_models_parameters_aux = list(distribution = distribution,
                                       param_now = param_now,
                                       path_to_save_model = path_to_save_model,
                                       one_hot_max_sizes = one_hot_max_sizes,
                                       use_cuda = use_cuda,
                                       epochs = epochs,
                                       num_different_vaeac_initiate = num_different_vaeac_initiate,
                                       epochs_initiation_phase = epochs_initiation_phase,
                                       masking_ratio = masking_ratio,
                                       validation_ratio = validation_ratio,
                                       validation_iwae_num_samples = validation_iwae_num_samples,
                                       width = width,
                                       depth = depth,
                                       latent_dim = latent_dim,
                                       lr = lr,
                                       batch_size = batch_size,
                                       running_avg_num_values = running_avg_num_values)
    
    # Add this to the explainer object
    explainer$VAEAC = list(models     = vaeac_models_aux,
                           results    = vaeac_models_results_aux,
                           parameters = vaeac_models_parameters_aux)
  }
  
  # Return the explainer
  return(explainer)
}

# An internal function that should in the final product not be called by the user.
# The user should rather call explain(...), which then will call on this function
# if the approach provided to explain(...) was vaeac.
# This function calls on prepare_data.vaeac function that creates the samples for
# each of observation for each coalition. 
# Then it calls on shapr:::prediction(...) which computes the Shapley values.
# Then throw away the imputed data and return the Shapley values.
explain.vaeac = function(x, explainer, approach, prediction_zero, which_vaeac_model = "best", verbose = FALSE, ...) {
  # Here x is the test data.
  # Some redundancy as we have to add which approach we want,
  # but as the explainer has much added stuff after been through 
  # 'add_vaeac_to_explainer()', the user shouldn't have to provide that
  # he/she will use vaeac approach. 
  
  # Check that we are provided with a vaeac model if we want to use that appraoch
  if (tolower(approach) == "vaeac" & is.null(explainer$user_provided_VAEAC)) {
    # We want to use the vaeac approach, but 'add_vaeac_to_explainer()' has not
    # been called.
    stop("Want to use the VAEAC approach, but no VAEAC-model was provided.
         The 'explainer' to this function needs to be the output of 
         the function 'add_vaeac_to_explainer()' which fits/trains a VAEAC.")
  }
  
  # Check validity of user input
  if (explainer$user_provided_VAEAC == TRUE & which_vaeac_model != 'best') {
    which_vaeac_model = 'best'
    
    warning("The VAEAC-model was provided by the user. 
             Cannot choose antoher VAEAC-model. 
             Fall back to the provided VAEAC-model.")
  }
  
  # Check validity of user input
  if (!(which_vaeac_model %in% c('best', 'running_best', 'last'))) {
    which_vaeac_model = 'best'
    
    warning("Do not recognise which VAEAC-model to use. 
            Can only be 'best', 'running_best', or 'last'.
            Fall back to 'best', and continue the computaions.")
  }
  
  # If data set contains categorical values we need to update the
  # data.table so assert that we have the right levels for each factors.
  categorical_col = unname(explainer$feature_list$classes == "factor")
  categorical_in_data_set = sum(categorical_col) > 0
  if(categorical_in_data_set) {
    explainer$x_test = preprocess_data(x, explainer$feature_list)$x_dt
  } else {
    explainer$x_test = as.matrix(preprocess_data(x, explainer$feature_list)$x_dt)
  }
  
  # Save that we are doing the vaeac approach
  explainer$approach = approach
  # Save the path to the VAEAC model that we will use
  explainer$VAEAC$path = explainer$VAEAC$models[[which_vaeac_model]]
  
  ### Generate data
  # Here we would use the generic 'prepare_data()' function
  # dt = prepare_data(explainer, ...)
  # which would then call on the 'prepare_data.vaeac' function.
  # However, as this code is currently no a part of the R-package SHPR.
  # We cannot do this, as the function would then have to be in the
  # right namespace. So we rather call 'prepare_data.vaeac' directly.
  dt = prepare_data.vaeac(explainer, verbose = verbose, ...)
  
  # If we are only returning the generated data, 
  # before computing shapley values.
  if (!is.null(explainer$return)) {
    return(dt)
  }
  
  ### Predict
  # Use the generated data to compute the shapley values
  # Here we need to specify that we are using shapr:::prediction.
  # When this code is included in the shapr library, we do not need
  # to specify shapr as these functions would then be in the same 
  # namespace and would be able to 'see each other'.
  r = suppressWarnings(shapr:::prediction(dt = dt, prediction_zero, explainer))
  
  # Return the computed shapley values together with the model
  # and some extra stuff.
  return(r)
}


# This is an internal function that should not be directly called by the user.
# It should only be called from the 'explain.vaeac()' function.
# This function samples data from a VAEAC stored in the explainer object.
# Originally called 'x' in this function in shapr package.
prepare_data.vaeac = function(explainer, seed = 1996, n_samples = 1e3L, index_features = NULL, verbose = TRUE, ...) {
  # Do not currently support 'index_features'.

  # These are additional column names in the data table with the 
  # imputed instances that we return in the end
  id = id_combination = w = NULL # due to NSE notes in R CMD check
  
  # Get the number of test observations
  n_xtest = nrow(explainer$x_test)
  
  # Create a list that will store the 
  # imputed/sampled observations from VAEAC
  dt_l = list()
  
  # Set seed for reproducibility
  if (!is.null(seed)) set.seed(seed)
  
  # index_features IS CURRENTLY NOT SUPPORTED
  if (is.null(index_features)) {
    features = explainer$X$features
  } else {
    features = explainer$X$features[index_features]
  }
  
  # Get number of coalitions and total number of explanatory
  # variables, not only those in index_features.
  n_coalitions = nrow(explainer$S)
  n_variables = ncol(explainer$S)
  
  # VAEAC impute the missing values by taking them in as NaN.
  # So the given variables are numbers and the dependent variables are NaN.
  # To know which variables to replace with NaN in each coalition we use 
  # the S matrix in the explainer object. S is a matrix where S_{ij} is 1 if
  # the j'th variable in the i'th coalition is independent, and 0 if it is dependent.
  # 1 -> given, 0 -> unknown and needs to be estimated. 
  # So all values that are 0 we replace with NaN.
  # This 'mask' matrix can then be elementwise multiplied with 
  # repeated versions of the test data/observations to create all 
  # coalitions and the instances for which the vaeac should estimate the 
  # missing values.
  mask = explainer$S
  mask[mask == 0] = NaN
  
  # Create a local version here of explainer$x_test that handels 
  # categorical values and all of that. 
  # If dataset contains categorical values we need to update the
  # data.table so assert that we have the right levels for each factors.
  categorical_col = unname(explainer$feature_list$classes == "factor")
  categorical_in_data_set = sum(categorical_col) > 0
  
  if (categorical_in_data_set) {
    # Data table contains categorical variables
    
    # Convert data table to matrix
    x_test_updated = matrix(apply(explainer$x_test, 2, as.numeric),
                            ncol = length(categorical_col))

    # Get the indices of the columns that are categorical
    categorical_col_ind = seq(length(categorical_col))[categorical_col]
    
    # Loop over the categorical columns at make sure that 
    # different classes are 0, 1, ..., K-1, if there is K classes
    # for the specific category.
    idx = 1
    for (idx in categorical_col_ind) {
      
      # Get the current categorical column
      col_now = x_test_updated[,idx]
      
      #levels_data = sort(unique(as.character(col_now)))
      levels_data = explainer$feature_list$factor_levels[[idx]]
      new_levels = explainer$feature_list$factor_levels_updated[[explainer$feature_list$labels[idx]]]

      col_new = rep(NA, length(col_now))
      for (old_level_idx in seq(length(explainer$feature_list$factor_levels[[idx]]))) {
        old_level = explainer$feature_list$factor_levels[[idx]][old_level_idx]
        col_new[col_now == as.integer(old_level)] = new_levels[old_level_idx]
      }
      col_new

      x_test_updated[,idx] = col_new
    }
  } else {
    # Data table contains only numerical variables
    x_test_updated = as.matrix(explainer$x_test)
  }
  
  x_test_updated

  # Treat each individual in the training data separately.
  # This loop could be parallelized, but would then load the
  # VAEAC model several times. 
  for (i in seq(n_xtest)) {
    if (verbose) {
      if (i %in% seq(10) || i%%(n_xtest/10) == 0) {
        cat(sprintf("Computing imputations for test observation %d of %d.\n", i, n_xtest))
      }
    }
    
    # Get the current test instance
    data_instance_i = x_test_updated[i,]
    
    # Create the num_coaltions different versions of this observation, 
    # where some of the variables are masked/NaN depending on the coalition.
    data_instance_i = matrix(data_instance_i, byrow = TRUE, 
                             ncol = n_variables, nrow = n_coalitions)
    
    # Apply the mask by elementwise multiplication to get all possible coalitions
    data_instance_i = data_instance_i * mask
    
    # Remove the first and last coalition (all variables unknown or known),
    # as these coalitions are not used to compute the shapley values.
    data_instance_i = data_instance_i[-c(1, n_coalitions), , drop=FALSE]
    

    # Compute num_imputations for all 2^p-2 coalitions for the i'th instance.
    # By using the VAEAC-model saved in the explainer object.
    VAEAC_conditional_data = VAEAC_impute_values(instances_to_impute = data_instance_i,
                                                 path_VAEAC_model = explainer$VAEAC$path,
                                                 num_imputations = as.integer(n_samples),
                                                 use_cuda = explainer$VAEAC$parameters$use_cuda,
                                                 one_hot_max_sizes = explainer$VAEAC$parameters$one_hot_max_sizes,
                                                 verbose = explainer$VAEAC$parameters$verbose_summary)
    
    aux = apply(apply(VAEAC_conditional_data, c(3,1), identity), 2, identity)
    id_combination_array = as.integer(c(1, rep(seq(2, n_coalitions-1), each = n_samples), n_coalitions))
    aux3 = data.table(id_combination = id_combination_array,
                      rbind(x_test_updated[i,], 
                            aux, 
                            x_test_updated[i,]))

    # Set the column-names
    setnames(aux3, c("id_combination", explainer$feature_list$labels))
    
    # Save the data for the i'th test observation in the dt list 
    dt_l[[i]] = aux3
    
    # Add a column with the sampling weights. 
    # This is just 1/n_samples in the case of VAEAC 
    # as we can sample as many instances as we will, 
    # in contrast to the emperical and ctree approach.
    dt_l[[i]][, w := 1]
    dt_l[[i]][-c(1, nrow(dt_l[[i]])), w := 1 / n_samples]
    
    # Then we add an id to the rows to show that all of these
    # samples belong to the i'th test observation.
    dt_l[[i]][, id := i]
    
    # Not supported!
    if (!is.null(index_features)) dt_l[[i]][, id_combination := index_features[id_combination]]
  }
  
  # Convert the list of data tables into one gigantic/long data table.
  dt = data.table::rbindlist(dt_l, use.names = TRUE, fill = TRUE)

  if (categorical_in_data_set) {
    categorical_col_ind = seq(length(categorical_col))[categorical_col]

    for (idx in categorical_col_ind) {
      col_now = dt[[explainer$feature_list$labels[idx]]]
      levels_now = explainer$feature_list$factor_levels_updated[[explainer$feature_list$labels[idx]]]
      levels_original = explainer$feature_list$factor_levels[[explainer$feature_list$labels[idx]]]

      col_new = rep(NA, length(col_now))
      for (level_now_idx in seq(length(explainer$feature_list$factor_levels[[idx]]))) {
        level_now = levels_now[level_now_idx]
        col_new[col_now == as.integer(level_now)] = levels_original[level_now_idx]
      }
      col_new = factor(col_new)

      dt[[explainer$feature_list$labels[idx]]] = col_new
    }
  }

  # Return the full data table.
  return(dt)
}

