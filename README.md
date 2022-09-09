# Shapley values and the VAEAC method

In this GitHub repository, we present the implementation of our ![equation](https://latex.codecogs.com/svg.latex?\texttt{VAEAC}) and ![equation](https://latex.codecogs.com/svg.latex?\texttt{VAEAC}_\mathcal{C}) approaches to Shapley value estimation used in our article "Using Shapley Values and Variational Autoencoders to Explain Predictive Models with Dependent Mixed Features", see [Olsen et al. (2022)](https://www.jmlr.org/papers/volume23/21-1413/21-1413.pdf). 

The variational autoencoder with arbitrary condiditioning (![equation](https://latex.codecogs.com/svg.latex?\texttt{VAEAC})) approach is based on the work of (Ivanov et al., 2019) and we extend it to ![equation](https://latex.codecogs.com/svg.latex?\texttt{VAEAC}_\mathcal{C}) which employs a simple but effective masking scheme to Shapley value estimation in sampled high-dimensions, see [Olsen et al. (2022)](https://www.jmlr.org/papers/volume23/21-1413/21-1413.pdf). The ![equation](https://latex.codecogs.com/svg.latex?\texttt{VAEAC}) is an extension of the regular variational autoencoder (Kingma and Welling, 2019). Instead of giving a probabilistic representation for the distribution ![equation](https://latex.codecogs.com/svg.latex?p(\boldsymbol{x})) it gives a representation for the conditional distribution ![equation](https://latex.codecogs.com/svg.latex?p(\boldsymbol{x}_{\mathcal{S}}&space;\mid&space;\boldsymbol{x}_{\bar{\mathcal{S}}})), for all possible feature subsets ![equation](https://latex.codecogs.com/svg.latex?\mathcal{S}\subseteq\mathcal{M}) simultaneously, where ![equation](https://latex.codecogs.com/svg.latex?\mathcal{M}) is the set of all features. The ![equation](https://latex.codecogs.com/svg.latex?\texttt{VAEAC}_\mathcal{C}) method focus on a specified set ![equation](https://latex.codecogs.com/svg.latex?\mathcal{C}) of coaltions sampled from ![equation](https://latex.codecogs.com/svg.latex?\mathcal{P}(\mathcal{M})), i.e., the set of all possible coalitions.

To make the ![equation](https://latex.codecogs.com/svg.latex?\texttt{VAEAC}) methodology work in the Shapley value framework, established in the R-package [`Shapr`](https://github.com/NorskRegnesentral/shapr) (Sellereite and Jullum, 2019), we have made alterations to the [original implementation](https://github.com/tigvarts/vaeac) of Ivanov.

The ![equation](https://latex.codecogs.com/svg.latex?\texttt{VAEAC}) and ![equation](https://latex.codecogs.com/svg.latex?\texttt{VAEAC}_\mathcal{C}) methods are implemented in Pytorch, hence, that portion of the repository is written in Python.
To compute the Shapley values, we have written the necessary R-code to make the ![equation](https://latex.codecogs.com/svg.latex?\texttt{VAEAC}) and ![equation](https://latex.codecogs.com/svg.latex?\texttt{VAEAC}_\mathcal{C}) approaches run on top of the R-package `shapr`.


## Setup

In addition to the prerequisites required by [Ivanov](https://github.com/tigvarts/vaeac/blob/master/requirements.txt), we also need several R-packages. All prerequisites are specified in `requirements.txt`. 

This code was tested on Linux and macOS (should also work on Windows), Python 3.6.4, PyTorch 1.0. and R 4.0.2.

The user has to specify the system path to the Python environment and the system path of the downloaded repository in `Source_Shapr_VAEAC.R`.



## Example

The following example shows how a random forest model is trained on the *Abalone* data set from the UCI machine learning repository, and how `shapr` explains the individual predictions.

Note that we only use **Diameter** (continuous), **ShuckedWeight** (continuous), and **Sex** (categorical) as features and let the response be **Rings**, that is, the age of the abalone.


``` r
# Import libraries
library(shapr)
library(ranger)
library(data.table)

# Load the R files needed for computing Shapley values using VAEAC.
source("ShapleyValuesVAEAC/Source_Shapr_VAEAC.R")

# Set the working directory to be the root folder of the GitHub repository. 
setwd("ShapleyValuesVAEAC")

# Read in the Abalone data set.
abalone = readRDS("data/Abalone.data")
str(abalone)

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
plot(explanation, plot_phi0 = FALSE)
```

<img src="figures/Vignette_results.png" width="100%" />




Then we look at how we can use ![equation](https://latex.codecogs.com/svg.latex?\texttt{VAEAC}_\mathcal{C}) in this too simple set up. 
``` r
# A small illustration of the use of VAEAC when Shapr samples the coalitions. (samples 4 coalitions)
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
plot(explanation_sampled, plot_phi0 = FALSE)
```

<img src="figures/Vignette_results.png" width="100%" />


## Citation

If you find this code useful in your research, please consider citing our paper:
```
@article{olsen2022using,
  title={Using Shapley Values and Variational Autoencoders to Explain Predictive Models with Dependent Mixed Features},
  author={Olsen, Lars Henry Berge and Glad, Ingrid Kristine and Jullum, Martin and Aas, Kjersti},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={213},
  pages={1--51},
  year={2022}
}
```

## References

Ivanov,  O.,  Figurnov,  M.,  and  Vetrov,  D.  (2019).  “Variational  Autoencoder  with  ArbitraryConditioning”. In:International Conference on Learning Representations.

Kingma, D. P. and Welling, M. (2014). "Auto-Encoding Variational Bayes". In: 2nd International Conference on Learning Representations, ICLR 2014.

Olsen, L. H. B., Glad, I. K., Jullum, M. and Aas, K. (2022). "Using Shapley Values and Variational Autoencoders to Explain Predictive Models with Dependent Mixed Features".

Sellereite,  N.  and  Jullum,  M.  (2019).  “shapr:  An  R-package  for  explaining  machine  learningmodels with dependence-aware Shapley values”. In:Journal of Open Source Softwarevol. 5,no. 46, p. 2027.

