# FuzzyClusteringSimilarity

[![Build Status](https://github.com/ryandewolfe33/FuzzyClusteringSimilarity.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ryandewolfe33/FuzzyClusteringSimilarity.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ryandewolfe33/FuzzyClusteringSimilarity.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ryandewolfe33/FuzzyClusteringSimilarity.jl)

A collection of method for computing similarity scores (also called a similarity index) between two (possibly fuzzy) clusterings. Indexes are typically in the range [0, 1], but many random clusterings produce values very close to 1 making them difficult to interpret. An Adjusted Similarity Index is computed using the following equation:
$$ \text{Adjusted Index} = \frac{\text{Index} - \mathbb{E}[\text{Index}]}{\max(\text{Index}) - \mathbb{E}[\text{Index}]}$$
Adjusted Indexes are in the range $(-\infty, 1)$ and an adjusted index of $0$ means that the clusterings are no more similar than expected if the clusterings had been selected at random. A popular adjusted similarity index is the Adjusted Rand Index (ARI). 

Computing an adjusted similarity index requries choices of both a score and a random model. This package provides implementations of several indexes and random models that are extensions of the Rand Index to fuzzy clusterings. The choice of a particular index and random model is problem dependent. See [[1]](https://doi.org/10.1007/s11634-025-00625-w) and [[2]](https://www.jmlr.org/papers/v18/17-039.html) for a discussion of this selection.

The following indexes and random models are implemented. If you use any of these indices please consider citing the paper where they were introduced.

### Indexes
- Normalized Degree of Concordance (NDC) [[3]](https://doi.org/10.1109/TFUZZ.2011.2179303)
- Frobenious [[4]](https://doi.org/10.1007/s00357-021-09407-3)
- Jousselme [[5]](https://doi.org/10.1109/TFUZZ.2017.2718484)
- Belief [[5]](https://doi.org/10.1109/TFUZZ.2017.2718484)
- Consistency [[5]](https://doi.org/10.1109/TFUZZ.2017.2718484)

### Random Models
- Permutation 
- Fit [[1]](https://doi.org/10.1007/s11634-025-00625-w)
- Sym [[1]](https://doi.org/10.1007/s11634-025-00625-w)
- Flat [[1]](https://doi.org/10.1007/s11634-025-00625-w)

Using the NDC and permutation model is called the Adjusted Degree of Concordance [[6]](https://doi.org/10.1007/s00357-020-09367-0). Note that the Frobenious Index may only be adjusted with the permutation model (see [[4]](https://doi.org/10.1007/s00357-021-09407-3) for details).

## Getting Started

This package is available from the julia general repository.

```julia
using Pkg
Pkg.add("FuzzyCLusteringSimilarity")
```

Then import the module.
```julia
using FuzzyClusteringSimilarity
```

You can run the unit tests to insure the package was properly installed.
```julia
Pkg.test("FuzzyClusteringSimilarity")
```

# Documentation
The package exports three main functions. It also exports a typing system of similarity scores and random models to be used to direct the function to the desired implementation. For an example of using the package, [check out the code](https://github.com/ryandewolfe33/DirichletRandAdjustmentModels) that produced the figures in [1].

```julia
function adjustedsimilarity(
    z1::AbstractMatrix{<:Real},
    z2::AbstractMatrix{<:Real},
    index::AbstractIndex,
    model::AbstractRandAdjustment;
    onesided::Bool=true
)
```
This is the main function to be used to compare two fuzzy clusterings, z1 and z2. The clusterings are in the form of a c x n matrix of n objects into c clusters. If the clustering is hard, ensure the matrix entries have type <:INT to call the proper random model.

```julia
function similarity(
    z1::AbstractMatrix{<:Real},
    z2::AbstractMatrix{<:Real},
    index::AbstractIndex
)
```
The similarity function can be used to compute and unadjusted index.

```julia
function expectedsimilarity(
    z1::AbstractMatrix,
    z2::AbstractMatrix,
    index::AbstractIndex,
    model::AbstractRandAdjustment;
    onesided::Bool=true
)
```
The expected similarity function computes the expected similarity index between random clusterings (using the provided random model).

```julia
function massageMatrix(matrix::AbstractMatrix)
```
Massage a matrix to enable julia's multiple dispatch. Matrix is formated with objects as columns and clusters and rows. If matrix is a hard clustering, the type is converted to Bool.

# References

[1] DeWolfe, R., Andrews, J.L. Random models for adjusting fuzzy rand index extensions. Adv Data Anal Classif (2025). https://doi.org/10.1007/s11634-025-00625-w

[2] Gates AJ, Ahn Y-Y (2017) The impact of random models on clustering similarity. J Mach Learn Res 18(87):1–28. http://jmlr.org/papers/v18/17-039.html

[3] E. Hullermeier, M. Rifqi, S. Henzgen and R. Senge, "Comparing Fuzzy Partitions: A Generalization of the Rand Index and Related Measures," in IEEE Transactions on Fuzzy Systems, vol. 20, no. 3, pp. 546-556, (2012). https://doi.org/10.1109/TFUZZ.2011.2179303

[4] Andrews, J.L., Browne, R. & Hvingelby, C.D. On Assessments of Agreement Between Fuzzy Partitions. J Classif 39, 326–342 (2022). https://doi.org/10.1007/s00357-021-09407-3

[5] T. Denoux, S. Li, and S. Sriboonchitta, “Evaluating and Comparing Soft Partitions: An Approach Based on Dempster–Shafer Theory,” IEEE Trans. Fuzzy Syst., vol. 26, no. 3, pp. 1231–1244, (2018), https://doi.org/10.1109/TFUZZ.2017.2718484.

[6] D’Ambrosio, A., Amodio, S., Iorio, C. et al. Adjusted Concordance Index: an Extensionl of the Adjusted Rand Index to Fuzzy Partitions. J Classif 38, 112–128 (2021). https://doi.org/10.1007/s00357-020-09367-0
