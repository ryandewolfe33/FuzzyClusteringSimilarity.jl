#= Based on the paper "Comparing Fuzzy Partitions: A Generalization of the Rand Index and Related Measuers"
α = 1 - |ui - uj|
disc = |α^U(i,j) - α^V(i,j)|
=#

using LinearAlgebra

struct NDC <: AbstractIndex
    p::Integer
    q::Real
    function NDC(p, q)
        p => 0 ? DomainError(p, "value for norm must be non-negative integer.") :
             q > 0 ? DomainError(q, "value for exponent must be positive") : new(p, q)
    end
end


function agreement(ui::Vector{<:Real}, uj::Vector{<:Real}, index::NDC)
    <:Real
    return 1 - norm(ui - uj, index.p)
end

function discordance(agreement1::Real, agreement2::Real, index::NDC)
    <:Real
    return abs(agreement1 - agreement2)^index.p
end