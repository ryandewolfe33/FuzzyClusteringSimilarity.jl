module FuzzyClusteringSimilarity

using Distributions
using SpecialFunctions

include("Rand/Index.jl")
include("Rand/Adjustment.jl")
include("MassageMatrix.jl")

function adjustedsimilarity(z1::AbstractMatrix{<:Real}, z2::AbstractMatrix{<:Real},
        index::AbstractIndex, model::AbstractRandAdjustment; onesided::Bool=true)
    # TODO add nsamples control
    expected = expectedsimilarity(z1, z2, index, model, onesided=onesided)
    if expected > 0.99
        @warn "Expected similarity is $(expected), results could be unstable."
    end
    return (similarity(z1, z2, index) - expected) / (1 - expected)
end

function adjustedsimilarity(zs::Vector{AbstractMatrix{<:Real}}, groundtruth::AbstractMatrix{<:Real}, index::AbstractIndex, model::AbstractRandAdjustment)
    similarities = Vector{Float64}(undef, length(zs))
    # TODO parallel flag
    for i in eachindex(zs)
        z = zs[i]
        similarities[i] = adjustedsimilarity(z, groundtruth, index, model, onesided=true)
    end
    return similarities
end

function adjustedsimilarity(zs::Vector{AbstractMatrix{<:Real}}, index::AbstractIndex, model::AbstractRandAdjustment)
    nmats = length(zs)
    similarities = Matrix{Float64}(undef, (nmats, nmats))
    for i in 1:nmats
        for j in 1:i
            ai = adjustedsimilarity(zs[i], zs[j], index, model, onesided=false) #Two sided when no ground truth
            similarities[i, j] = similarities[j, i] = ai
        end
    end
    return similarities
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