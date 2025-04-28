#= Based on the paper "Comparing Fuzzy Partitions: A Generalization of the Rand Index and Related Measuers"
α = 1 - |ui - uj|
disc = |α^U(i,j) - α^V(i,j)|
=#

using LinearAlgebra

struct NDC <: AbstractAgreementConcordanceIndex
    p::Integer
    q::Real
    function NDC(p, q)
        p < 0 ? throw(DomainError(p, "value for norm must be non-negative integer.")) :
        q <= 0 ? throw(DomainError(q, "value for exponent must be positive")) : new(p, q)
    end
end

function NDC()
    return NDC(1, 1)
end

function agreement(ui::Vector{<:Real}, uj::Vector{<:Real}, index::NDC)::Real
    return 1 - norm(ui - uj, index.p) / 2^(1 / index.p)  # Scaling to get agreement between 0 and 1
end

function discordance(agreement1::Real, agreement2::Real, index::NDC)::Real
    return abs(agreement1 - agreement2)^index.q
end