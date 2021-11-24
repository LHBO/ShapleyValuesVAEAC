# Import libraries
library(shapr)
library(ranger)
library(data.table)

# Load the R files needed for computing Shapley values using VAEAC.
source("/Users/larsolsen/Desktop/PhD/R_Codes/Source_Shapr_VAEAC.R")

# Set the working directory to be the root folder of the GitHub repository. 
setwd("~/PhD/Paper1/Code_for_GitHub")

# Read in the Abalone data set.
abalone = readRDS("data/Abalone.data")
str(abalone)

# Seet seed
set.seed(2021)

# Predict rings based on Diameter, ShuckedWeight, and Sex (categorical), using a random forrest model.
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

# Computing the actual Shapley values with kernelSHAP accounting for feature dependence using
# the VAEAC distribution approach with parameters defined above
explanation = explain.vaeac(abalone[abalone$test_instance == TRUE][1:8,c(3,6,9)],
                            approach = "vaeac",
                            explainer = explainer_added_vaeac,
                            prediction_zero = phi_0,
                            which_vaeac_model = "best")

# Printing the Shapley values for the test data.
# For more information about the interpretation of the values in the table, see ?shapr::explain.
print(explanation$dt)
#>        none   Diameter  ShuckedWeight        Sex
#> 1: 9.927152  0.63282471     0.4175608  0.4499676
#> 2: 9.927152 -0.79836795    -0.6419839  1.5737014
#> 3: 9.927152 -0.93500891    -1.1925897 -0.9140548
#> 4: 9.927152  0.57225851     0.5306906 -1.3036202
#> 5: 9.927152 -1.24280895    -1.1766845  1.2437640
#> 6: 9.927152 -0.77290507    -0.5976597  1.5194251
#> 7: 9.927152 -0.05275627     0.1306941 -1.1755597
#> 8: 9.927153  0.44593977     0.1788577  0.6895557

# Finally, we plot the resulting explanations.
png("Vignette_results2.png", res = 150, height = 1000, width = 1250)
plot(explanation, plot_phi0 = FALSE)
dev.off()




order(explanation$p[ c(1, 2, 3, 4, 5, 6, 14, 16, 39, 48)])
1, 2, 3, 4, 5, 6, 14, 16, 39, 48

new_order = c(1, 2, 3, 4, 5, 6, 14, 16, 39, 48,  (1:100)[!(1:100 %in% c(1, 2, 3, 4, 5, 6, 14, 16, 39, 48))])

abalone2 = abalone
abalone3 = abalone2[abalone2$test_instance == TRUE]
abalone4 = abalone3[new_order]
abalone[abalone2$test_instance == TRUE] = abalone4

explanation$x


