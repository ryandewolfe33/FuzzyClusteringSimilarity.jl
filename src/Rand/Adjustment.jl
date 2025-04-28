abstract type AbstractRandAdjustment end

abstract type AbstractAgreementConcordance <: AbstractRandAdjustment end

function expectedsimilarity(z1::AbstractMatrix, z2::AbstractMatrix,
        index::AbstractIndex, model::AbstractRandAdjustment; onesided::Bool=true)
    throw(TypeError(
        :expectedsimilarity, "", AbstractRandAdjustment, model))
end

for fname in [
    "Permutation.jl",
    "DirichletModel.jl"
]
    include(joinpath("Adjustment", fname))
end