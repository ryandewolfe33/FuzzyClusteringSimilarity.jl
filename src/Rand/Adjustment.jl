abstract type AbstractRandAdjustment end

abstract type AbstractAgreementConcordance <: AbstractRandAdjustment end

function expectedindex(z1::AbstractMatrix, z2::AbstractMatrix,
        index::AbstractIndex, model::AbstractRandAdjustment)
    throw(TypeError(
        model, "Model is not defined not defined", AbstractRandAdjustment, type(model)))
end

for fname in [
    "Permutation.jl",
    "DirichletModel.jl"
]
    include(joinpath("Adjustment", fname))
end