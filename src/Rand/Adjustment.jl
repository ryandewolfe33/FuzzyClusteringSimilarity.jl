abstract type AbstractRandAdjustment end

abstract type AbstractAgreementConcordance<:AbstractAgreementConcordance end

for fname in [
    "Permutation.jl",
]
    include(joinpath("Adjustment", fname))
end