# Import libraries
library(shapr)
library(ranger)
library(data.table)

# Load the R files needed for computing Shapley values using VAEAC.
source("Source_Shapr_VAEAC.R")

# Read in the Abalone data set.
abalone = readRDS("data/Abalone.data")
str(abalone)

# Set seed
set.seed(2021)

# Predict rings based on Diameter, ShuckedWeight, and Sex (categorical), using a random forest model.
model = ranger(Rings ~ Diameter + ShuckedWeight + Sex, data = abalone[abalone$test_instance == FALSE,])

# Specifying the phi_0, i.e. the expected prediction without any features.
phi_0 <- mean(abalone$Rings[abalone$test_instance == FALSE])

# Prepare the data for explanation. Diameter, ShuckedWeight, and Sex correspond to 3,6,9.
explainer <- shapr(abalone[abalone$test_instance == FALSE, c(3,6,9)], model)
#> The specified model provides feature classes that are NA. The classes of data are taken as the truth.

# Train the VAEAC model with specified parameters and add it to the explainer
explainer_added_vaeac = add_vaeac_to_explainer(
  explainer, 
  epochs = 30L,
  width = 32L,
  depth = 3L,
  latent_dim = 8L,
  lr = 0.002,
  num_different_vaeac_initiate = 2L,
  epochs_initiation_phase = 2L,
  validation_iwae_num_samples = 25L,
  verbose_summary = TRUE)

# Compute the Shapley values with respect to feature dependence using
# the VAEAC_C approach with parameters defined above
explanation = explain.vaeac(abalone[abalone$test_instance == TRUE][1:8,c(3,6,9)],
                            approach = "vaeac",
                            explainer = explainer_added_vaeac,
                            n_samples = 250L,
                            prediction_zero = phi_0,
                            which_vaeac_model = "best")

# Printing the Shapley values for the test data.
# For more information about the interpretation of the values in the table, see ?shapr::explain.
print(explanation$dt)
#>        none   Diameter ShuckedWeight        Sex
#> 1: 9.927152  0.5514675     0.4102614  0.5386242
#> 2: 9.927152 -0.8691068    -0.5059807  1.5084370
#> 3: 9.927152 -1.1324510    -1.0110522 -0.8981503
#> 4: 9.927152  0.4321455     0.5323742 -1.1651909
#> 5: 9.927152 -1.4529236    -0.9864594  1.2636536
#> 6: 9.927152 -0.8819458    -0.5280294  1.5588355
#> 7: 9.927152 -0.2511181     0.2441703 -1.0906742
#> 8: 9.927152  0.4005953     0.2119935  0.7017644

# Finally, we plot the resulting explanations.
png("Vignette_results.png", res = 150, height = 1000, width = 1250)
plot(explanation, plot_phi0 = FALSE)
dev.off()


###########################
##### Masking Schemes #####
###########################
# A small illustration of the use of VAEAC when Shapr samples the coalitions. (samples 4 coalitions)
set.seed(2022)
explainer_sampled <- shapr(abalone[abalone$test_instance == FALSE, c(3,6,9)], model, n_combinations = 4)
#> The specified model provides feature classes that are NA. The classes of data are taken as the truth.

# Number of coalitions sampled (removed edge cases)
n_coalitions = nrow(explainer_sampled$S) - 2

# Train the VAEAC_C model with specified parameters and coalitions and add it to the explainer
explainer_added_vaeac_sampled = add_vaeac_to_explainer(
  explainer_sampled, 
  epochs = 30L,
  width = 32L,
  depth = 3L,
  latent_dim = 8L,
  lr = 0.002,
  num_different_vaeac_initiate = 2L,
  epochs_initiation_phase = 2L,
  validation_iwae_num_samples = 25L,
  verbose_summary = TRUE,
  mask_generator_only_these_coalitions = explainer_sampled$S[2:(n_coalitions+1),],
  mask_generator_only_these_coalitions_probabilities = explainer_sampled$X$shapley_weight[2:(n_coalitions+1)] # Do not need to be standardized. 
)

# Compute the Shapley values with respect to feature dependence using
# the VAEAC_C approach with parameters defined above
explanation_sampled = explain.vaeac(abalone[abalone$test_instance == TRUE][1:8,c(3,6,9)],
                                approach = "vaeac",
                                explainer = explainer_added_vaeac_sampled,
                                n_samples = 250L,
                                prediction_zero = phi_0,
                                which_vaeac_model = "best")

# Printing the Shapley values based on the sampled coalitions for the test data.
print(explanation_sampled$dt)
#>        none    Diameter ShuckedWeight         Sex
#> 1: 9.927152  1.10051882     0.4878096 -0.08797535
#> 2: 9.927152 -0.57753158    -0.4496048  1.16048591
#> 3: 9.927152 -1.47832493    -0.9956322 -0.56769640
#> 4: 9.927152  0.98457292     0.2895466 -1.47479065
#> 5: 9.927152 -1.17077734    -0.8066726  0.80172046
#> 6: 9.927152 -0.60118281    -0.4358604  1.18590353
#> 7: 9.927152 -0.02444789     0.1094176 -1.18259171
#> 8: 9.927152  1.17234737     0.2177909 -0.07578500

# Finally, we plot the resulting explanations based on the sampled coalitions.
png("Vignette_results_sampled.png", res = 150, height = 1000, width = 1250)
plot(explanation_sampled, plot_phi0 = FALSE)
dev.off()






