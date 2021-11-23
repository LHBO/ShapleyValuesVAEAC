# Shapley values and the VAEAC method

In this GitHub repository, we present the implementation of the `VAEAC` approach from our paper "Using Shapley Values and Variational Autoencoders to Explain Predictive Models with Dependent Mixed Features", see [Olsen et al. (2021)](http://arxiv.com). 

The variational autoencoder with arbitrary condiditioning (`VAEAC`) approach is based on the work of (Ivanov et al., 2019). The `VAEAC` is an extension of 
the regular variational autoencoder (Kingma and Welling, 2019). Instead of giving a probabilistic representation for the distribution ![equation](https://latex.codecogs.com/svg.latex?p(\boldsymbol{x})) it gives a representation for the conditional ![equation](https://latex.codecogs.com/svg.latex?p(\boldsymbol{x}_{\mathcal{S}}&space;\mid&space;\boldsymbol{x}_{\bar{\mathcal{S}}})), for all possible feature subsets 
![equation](https://latex.codecogs.com/svg.latex?\mathcal{S}\subseteq\mathcal{M}) simultaneously, where ![equation](https://latex.codecogs.com/svg.latex?\mathcal{M}) is the set of all features.

To make the `VAEAC` methodology work in the Shapley value framework, established in the R-package [`Shapr`](https://github.com/NorskRegnesentral/shapr) (Sellereite and Jullum, 2019), we have made alterations to the [original implementation](https://github.com/tigvarts/vaeac) of Ivanov.

The `VAEAC` model in implemented in Pytorch, hence, that portion of the repository is written in Python.
To compute the Shapley values, we have written necessary R-code to make the `VAEAC` approach run on top of the R-package `shapr`.


## Setup

In addition to the prerequisites required by [Ivanov](https://github.com/tigvarts/vaeac/blob/master/requirements.txt), we also need several R-packages. All prerequisites are specified in `requirements.txt`. 

This code was tested on Linux and macOS.

Python 3.6.4 and PyTorch 1.0.

Python:
numpy==1.16.2
pandas==0.24.2
Pillow==4.2.1
torch==1.0.1
torchvision==0.2.2
tqdm==4.31.1


R version 4.0.2 (2020-06-22)
Platform: x86_64-apple-darwin17.0 (64-bit)
Running under: macOS Catalina 10.15.7

R:
reticulate==1.18
abind==1.4-5
shapr==0.2.0

## Example

The following example shows how a simple xgboost model is trained using the Boston Housing Data, and how shapr explains the individual predictions.


`shapr` supports computation of Shapley values with any predictive model
which takes a set of numeric features and produces a numeric outcome.

The following example shows how a simple `xgboost` model is trained
using the *Boston Housing Data*, and how `shapr` explains the individual
predictions.

``` r
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
#>        none   Diameter ShuckedWeight        Sex
#> 1: 9.927152  0.63282471     0.4175608  0.4499676
#> 2: 9.927152 -0.79836795    -0.6419839  1.5737014
#> 3: 9.927152 -0.93500891    -1.1925897 -0.9140548
#> 4: 9.927152  0.57225851     0.5306906 -1.3036202
#> 5: 9.927152 -1.24280895    -1.1766845  1.2437640
#> 6: 9.927152 -0.77290507    -0.5976597  1.5194251
#> 7: 9.927152 -0.05275627     0.1306941 -1.1755597
#> 8: 9.927153  0.44593977     0.1788577  0.6895557

# Finally, we plot the resulting explanations.
plot(explanation, plot_phi0 = FALSE)
```

<img src="figures/Vignette_results.png" width="100%" />





## Citation

If you find this code useful in your research, please consider citing our paper:
```
@article{
  Olsen2021Shapley,
  title={Using Shapley Values and Variational Autoencoders to Explain Predictive Models with Mixed Features},
  author={Lars Henry Berge Olsen and Ingrid Kristine Glad and Kjersti Aas and Martin Jullum},    
  journal = {TO BE ADDED},
  volume = {TO BE ADDED},
  pages = {TO BE ADDED},
  year = {2021},
  issn = {TO BE ADDED},
  doi = {TO BE ADDED},
  url = {TO BE ADDED}
}
```

## References

Ivanov,  O.,  Figurnov,  M.,  and  Vetrov,  D.  (2019).  “Variational  Autoencoder  with  ArbitraryConditioning”. In:International Conference on Learning Representations.

Kingma, D. P. and Welling, M. (2014). "Auto-Encoding Variational Bayes". In: 2nd International Conference on Learning Representations, ICLR 2014.

Sellereite,  N.  and  Jullum,  M.  (2019).  “shapr:  An  R-package  for  explaining  machine  learningmodels with dependence-aware Shapley values”. In:Journal of Open Source Softwarevol. 5,no. 46, p. 2027.

