#= Based on the paper "Comparing Fuzzy Partitions: A Generalization of the Rand Index and Related Measuers"
α = 1 - |ui - uj|
disc = |α^U(i,j) - α^V(i,j)|
=#

using LinearAlgebra

struct NDC <: AbstractIndex
    p <: Integer
    q <: Real
end
function NDC(p = 1 <: Integer, q = 1 <: Real)
    if p < 0
        throw(DomainError(p, "p-norms only defined on non-negative integers"))
    end
    if q <= 0
        throw(DomainError(q, "q must be positive."))
    end
    NDC(p, q)
end

function agreement(ui::Vector{<:Real}, uj::Vector{<:Real}, index::NDC)
    <:Real
    return 1 - norm(ui - uj, index.p)
end

function discordance(agreement1 <: Real, agreement2 <: Real, index::NDC)
    <:Real
    return abs(agreement1 - agreement2)^index.p
end