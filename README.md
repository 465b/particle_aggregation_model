# particle aggregation model

## Overview

This model calculates particle aggregation based on discrete aggregate size classes within a volume (0D box). 
It is based on paper of [Jackson and Lochmann (1992)](https://doi.org/10.4319/lo.1992.37.1.0077) and [Adrian Burd (2013)](https://doi.org/10.1002/jgrc.20255).
This python implementation is based on Matlab code by Adrian Burd (found in `coag_model_adrian_matlab`) which was adapted from the code of George Jackson.

For a more detailed description and context read the [Jackson and Lochmann (1992) paper](https://doi.org/10.4319/lo.1992.37.1.0077), but the general idea goes as follows.
We would like to make predictions about the time evolution of a aggregate size distribution $\frac{dn(m,t)}{dt}$, where $n(m,t)$ is the number of particles of mass $m$ at time $t$

![Equation 1 from Jackson & Lochmann 1992](doc/figures/jl_eq1.png)

$alpha$ represents the sicking probability, $beta$ is referred to as the coagulation kernel - a measure of the probability of collisions between aggregates.

Integrating this directly proves difficult. To gain some traction we therefore discretize the aggreates into a discrete size spectrum.
To make the calculations easier a size bin is defined to range from their lower mass boundary to twice their lower mass boundary, $m_{lower}$ to $2m_{lower} = m_{upper}$.
Using this discretization equation 1 can be rewritten as:

![Equation 2 from Jackson & Lochmann 1992](doc/figures/jl_eq9.png)

$dQ_i/dt$ represents the change in mass of size bin $i$ 
$^{i} \beta$ are referred to sectional coagulation kernels. Each represents a different class of aggregate interactions (e.g. collisions between aggregates in the same size bin, collisions between aggregates in adjacent size bins, etc.) and is defined as:

![Table 1 - Sectional coagulation kernels](doc/figures/jl_sectional_kernel.png)

Based on the index ranges within the table, the interactions can be visualized as interactions between bins as follows:

![Aggregate size bin interactions](doc/figures/bin_interactions.png) (figure not yet included)


## To do
The model is currently not running as expected and produces in part non-physical results.
The calculation of the sectional coagulation functions is suspected to be incorrect.
In particular an indexing error seems to be present.

To side step this issue, we hardcoded the sectional coagulation functions from the matlab code for the 20 size classes case.
However, this seems to also produce declining total volume concentrations over time even tho the total volume concentration is expected to be conserved.

- [ ] Fix sectional coagulation functions
- [ ] Fix total volume concentration issue in time integration

## How to install

<!-- You can install the package using pip:

```bash
pip install particle_aggregation_model
``` -->

To install the package from source:

```bash
git clone https://github.com/465b/particle_aggregation_model.git
cd particle_aggregation_model
python -m build
pip install -e .
```

## How to run

To get started, take a look at the [demo notebookts](https://github.com/465b/particle_aggregation_model/tree/781442d2c50fe9cd2d1ae08610ba30fd2c887098/demos)m in particular [model_demo_and_validation.ipynb](https://github.com/465b/particle_aggregation_model/blob/609ce09448798ce18fd13ab20eea9e1059e9c839/demos/model_demo_and_validation.ipynb)
<!-- include the html of the notebook -->

## Get in touch

This is an early draft of the model. If you are curious to use it, or have any questions, please get in touch with me: laurin.steidle@uni-hamburg.de