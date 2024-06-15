module FuzzyClusteringSimilarity

using Distributions
using SpecialFunctions

include("Rand/Index.jl")
include("Rand/Adjustment.jl")
include("MassageMatrix.jl")

function adjustedindex(z1::AbstractMatrix{<:Real}, z2::AbstractMatrix{<:Real},
        index::AbstractIndex, model::AbstractRandAdjustment)
    # TODO add one vs two sided
    # TODO add nsamples control
    expected = expectedindex(z1, z2, index, model)
    return (index(z1, z2, index) - expected) / (1 - expected)
end

# Indexes
export AbstractIndex
# From DempsterShafer
export Jousseleme
export Belief
export Consistency
# From NDC
export NDC

# Adjustments
export AbstractRandAdjustment
export AbstractAgreementConcordance
# From DirichletModels
export FlatDirichlet
export SymmetricDirichlet
export FitDirichlet
# From Permutation
export Permutation

# functions
export index
export expectedindex
export adjustedindex

# Utils
export massageMatrix

end