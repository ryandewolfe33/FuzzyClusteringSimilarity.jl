abstract type AbstractIndex end

#Generic discordance to make compiler happy
function discordance(agreement1, agreement2, index<:AbstractIndex)
    throw(TypeError(discordance, "Discordance not defined for this index", AbstractIndex, type(index)))
end


function discordance(ui::Vector{<:Real}, uj::Vector{<:Real}, vi::Vector{<:Real}, vj::Vector{<:Real}, index<:AbstractIndex) <: Real
    agreement1 = agreement(ui, uj, index)
    agreement2 = agreement(vi, vj, index)
    return discordance(agreement1, agreement2, index)
end

for fname in [
    "DempsterShafer.jl",
    "NDC.jl"
]
    include(joinpath("Index", fname));
end