module FuzzyClusteringSimilarity

using Distributions
using SpecialFunctions

include("Rand/Index.jl")
include("Rand/Adjustment.jl")
include("MassageMatrix.jl")

function adjustedsimilarity(z1::AbstractMatrix{<:Real}, z2::AbstractMatrix{<:Real},
        index::AbstractIndex, model::AbstractRandAdjustment)
    # TODO add one vs two sided
    # TODO add nsamples control
    expected = expectedsimilarity(z1, z2, index, model)
    println(expected)
    println(similarity(z1, z2, index))
    return (similarity(z1, z2, index) - expected) / (1 - expected)
end

# Indexes
export AbstractIndex
# From DempsterShafer
export Jousseleme
export Belief
export Consistency
# From NDC
export NDC
# From Frobenious
export Frobenious

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
export similarity
export expectedsimilarity
export adjustedsimilarity

# Utils
export massageMatrix

end