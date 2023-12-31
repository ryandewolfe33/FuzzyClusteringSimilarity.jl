# FuzzyClusteringSimilarity

Code for Dirichlet Random Models for Fuzzy Rand Adjustment

[![Build Status](https://github.com/ryandewolfe33/FuzzyClusteringSimilarity.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ryandewolfe33/FuzzyClusteringSimilarity.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ryandewolfe33/FuzzyClusteringSimilarity.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ryandewolfe33/FuzzyClusteringSimilarity.jl)

## Summary
Adjusted Normalized Degree of Concordance (ANDC) is a similarity measure between two fuzzy (or hard) clusterings. The value is in `(-inf, 1]`, with 1 representing identical clusterings, and 0 representing the clusterings have the same agreement as "random" clusterings. The selection of "random" is required by the user, and four models are provided.

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

```julia
andc(matrix1::AbstractMatrix, matrix2::AbstractMatrix, model::String, oneSided=True, p::Int=1, q::Int=1)
```
Calculate the adjusted normalized degree of concordance of matrix1 and matrix2 using model to adjust for chance agreement. Available models are '"fit", "sym", "flat", "perm" '. 

```julia
ndc(matrix1::AbstractMatrix, matrix2::AbstractMatrix, p::Int=1, q::Int=1)
```
Calculate the normalized degree of concordance of matrix1 and matrix2.

```julia
endc(matrix1::AbstractMatrix, matrix2::AbstractMatrix, model::String, oneSided=True, p::Int=1, q::Int=1)
```
Calculate the expected normalized degree of concordance of random matrices. Models for generating random matrices based on matrix1 and matrix2  are '"fit", "sym", "flat", "perm" '.


```julia
massageMatrix(matrix::AbstractMatrix)
```
Massage a matrix to enable julia's multiple dispatch. Matrix is formated with points as columns and clusters and rows. If matrix is a hard clustering the type is converted to Bool.

# References

E. Hullermeier, M. Rifqi, S. Henzgen and R. Senge, "Comparing Fuzzy Partitions: A Generalization of the Rand Index and Related Measures," in IEEE Transactions on Fuzzy Systems, vol. 20, no. 3, pp. 546-556, June 2012. https://doi.org/10.1109/TFUZZ.2011.2179303

D’Ambrosio, A., Amodio, S., Iorio, C. et al. Adjusted Concordance Index: an Extensionl of the Adjusted Rand Index to Fuzzy Partitions. J Classif 38, 112–128 (2021). https://doi.org/10.1007/s00357-020-09367-0

Andrews, J.L., Browne, R. & Hvingelby, C.D. On Assessments of Agreement Between Fuzzy Partitions. J Classif 39, 326–342 (2022). https://doi.org/10.1007/s00357-021-09407-3

