## Infinite-horizon GPs: Matlab codes

We share a reference Matlab implementation of all methods considered in the **Infinite Horizon Gaussian processes** paper. The Matlab codes act as reference implementations but also feature debugging and other output not necessary in a practical implementation. Further, all codes are formulated from a batch processing point-of-view, even though they can be converted into streaming implementations with rather small modifications.

### Dependencies

* The codes have been tested under Matlab R2017b.
* For conversion between kernels (covariance functions) and state space models, we use code published as part of other toolboxes the authors have contributed to. Other `cf_[covariance function here]_to_ss.m` codes can be found, e.g., in the [GPstuff toolbox](https://research.cs.aalto.fi/pml/software/gpstuff/) or in the [GPML toolbox](http://www.gaussianprocess.org/gpml/code/matlab/doc/) in a more concise form.
* For non-Gaussian likelihoods, we leverage functions in the [GPML toolbox](http://www.gaussianprocess.org/gpml/code/matlab/doc/). Tested under version 4.2.

### Functions

See the function documentation `help [function name]` for details.

* `ihgpr.m` - Infinite-horizon GP regression.
* `ihgp_adf.m` - Infinite-horizon assumed density / single-sweep EP solution to GP models.
* `gp_solve.m` - Vanilla GP regression for comparison in the toy example.
* `gf_solve.m` - Filtering solution to temporal GP regression problems.
* `gf_adf.m` - Assumed density / single-sweep EP solution to GP models.

### Examples

See the codes and data under the corresponding subdirectory for details.

* `sinc` - The toy example from the paper
* `coal` - Coal mining example from the paper
* `classification` - Classification examples from the paper appendix
