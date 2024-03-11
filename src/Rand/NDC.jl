# Based on the paper "Comparing Fuzzy Partitions: A Generalization of the Rand Index and Related Measuers"

using LinearAlgebra

function ndc_agreement(ui::Vector{<:Real}, uj::Vector{<:Real}) <: Real
    return 1 - norm(ui - uj, 1)
end

function ndc_discordance(agreement1<:Real, agreement2<:Real) <: Real
    return norm(agreement1 - agreement2, 1) #Absolute value is just 1 norm of length 1 vectors. Easier to update to q norm later.
end

function ndc_discordance(ui::Vector{<:Real}, uj::Vector{<:Real}, vi::Vector{<:Real}, vj::Vector{<:Real}) <: Real
    agreement1 = ndc_agreement(ui, uj)
    agreement2 = ndc_agreement(vi, vj)
    return ndc_discordance(agreement1, agreement2)
end