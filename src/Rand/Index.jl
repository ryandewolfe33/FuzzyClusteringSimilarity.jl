abstract type AbstractIndex end

function index(z1::AbstractMatrix{<:Real}, z2::AbstractMatrix{<:Real}, index::AbstractIndex)
    if size(z1, 2) != size(z2, 2)
        throw(DimensionMismatch("Matrices must have same number of points along dimension 1. Got $(size(z1, 1)) and $(size(z2, 1))."))
    end
    @debug "Matrices of size $(size(z1)) and $(size(z2))."
    npoints = size(z1, 2)
    ncomparisons = npoints * (npoints - 1) / 2
    totaldiscordance = 0.0
    for i in 2:npoints
        for j in 1:(i - 1)
            totaldiscordance += discordance(z1[:, i], z1[:, j], z2[:, i], z2[:, j], index)
        end
    end
    @debug "Total discordance of $(totaldscordance) over $(ncomparisons) comparisons."
    return 1 - totaldiscordance / ncomparisons
end

#Generic error functions
function discordance(agreement1, agreement2, index::AbstractIndex)
    throw(TypeError(
        discordance, "Discordance not defined for this index", AbstractIndex, type(index)))
end

function agreement(ui, uj, index::AbstractIndex)::Real
    throw(TypeError(
        agreement, "Agreement not defined for this index.", AbstractIndex, type(index)))
end

# Generic Discordance function
function discordance(ui::Vector{<:Real}, uj::Vector{<:Real}, vi::Vector{<:Real},
        vj::Vector{<:Real}, index::AbstractIndex)::Real
    agreement1 = agreement(ui, uj, index)
    agreement2 = agreement(vi, vj, index)
    return discordance(agreement1, agreement2, index)
end

# Create a vector of agreements of the n(n-1)/2 comparisons from a zig matrix
function agreement(z1::AbstractMatrix{<:Real}, index::AbstractIndex)::Vector{Float64}
    npoints = size(z1, 2)
    ncomparisons = npoints * (npoints - 1) / 2
    zagreements = Vector{Float64}(undef, ncomparisons)
    indexcurrent = 1
    for i in 2:npoints
        for j in 1:(i - 1)
            zagreements[indexcurrent] = agreement(z1[:, i], z1[:, j], index)
            indexcurrent += 1
        end
    end
    return zagreements
end

for fname in [
    "DempsterShafer.jl",
    "NDC.jl"
]
    include(joinpath("Index", fname))
end
