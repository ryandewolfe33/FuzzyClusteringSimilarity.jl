# Based on the paper "Evaluating and Comparing Soft Partitions: An Approach Based on Depmster-Shafer Theory"
# Check out R package evclust

#=Prop 12, 
m_ij(∅) = 0
    since fuzzy clusterings are baysiean (all mass on singletons, m_i(∅) = 0)
m_ij(s) = ∑_k u_ik * u'_ik = u_i ⋅ u_j (dot product membership vectors)
m_ij(¬s) = 1 - m_ij(s)
    since A ∩ B = ∅ and m_i(A) * m_j(B) ≠ 0 only when A = {u_k}, B={u_m}, k ≠ m
m_ij(Θ) = 1 - m_ij(s) - m_ij(¬s) = 0
    Sum of all 4 cases must equal 1

Define similarity indices as agreement concordant rand like extensions
ρ_s(M, M') = 1 - E[δ(m_ij, m'_ij)] (reformulate equation 14)
where δ is a distance (discordance) between m_ij, m'_ij
=#

using LinearAlgebra

#=
Jousselme's Distance δJ 
See equations 15 in the paper
Define the Jaccard Matrix
J = [[1, 0, 0, 0],
    [0, 1, 0, 1/2],
    [0, 0, 1, 1/2],
    [0, 1/2, 1/2, 1]]

δJ = (0.5 * (m_ij - m'ij)^T * J * (m_ij - m'_ij))^(1/2)

So for clusterings A, B; vectors looks like [0, α^A_ij, 1-α^A_ij, 0] and [0, α^B_ij, 1-α^B_ij, 0] where agreement function α is the dot product

Then 
2 * δ_J(vectors) = [0, α^A_ij - α^B_ij, α^B_ij - α^A_ij, 0]^T * J * [0, α^A_ij - α^B_ij, α^B_ij - α^A_ij, 0]
                 = [0, α^A_ij - α^B_ij, α^B_ij - α^A_ij, 0]^T * [0, α^A_ij - α^b_ij, α^B_ij - α^A_ij, 0]
                 = (α^A_ij - α^B_ij)^2 + (α^B_ij - α^A_ij)^2
                 = 2(α^A - ij - α^B_ij)^2

So writing as agreement and discordance
α(i,j) = u_i ⋅ u_j
disc(i,j) = (α^A(i,j) - α^B(i,j))^2
=#
function jousselme_agreement(ui::Vector{<:Real}, uj::Vector{<:Real}) <: Real
    return dot(ui, uj)
end

function jousselme_discordance(agreement1<:Real, agreement2<:Real) <: Real
    return (agreement1 - agreement2)^2
end

function jousselme_discordance(ui::Vector{<:Real}, uj::Vector{<:Real}, vi::Vector{<:Real}, vj::Vector{<:Real}) <: Real
    agreement1 = jousseleme_agreement(ui, uj)
    agreement2 = jousseleme_agreement(vi, vj)
    return jousselme_discordance(agreement1, agreement2)
end

#=
Belief Distance δB
See equation 17

δB(vectors) = 1/2( |m(s) - m'(s)| + |m(¬s) - m'(¬s)| + |m(∅) - m'(∅)|
            = 1/2( |u_i ⋅ u_j - u'_i ⋅ u'_j| + |1 - u_i ⋅ u_j - 1 + u'_i ⋅ u'_j|)
            = |u_i ⋅ u_j - u'_i ⋅ u'_j|

So agreement and discordance are given by 
α(i,j) = u_i ⋅ u_j
disc(i,j) = |α^A(i,j) - α^B(i,j)|
=#

function belief_agreement(ui::Vector{<:Real}, uj::Vector{<:Real}) <: Real
    return dot(ui, uj)
end

function belief_discordance(agreement1<:Real, agreement2<:Real) <: Real
    return abs(agreement1 - agreement2)
end

function belief_discordance(ui::Vector{<:Real}, uj::Vector{<:Real}, vi::Vector{<:Real}, vj::Vector{<:Real}) <: Real
    agreement1 = belief_agreement(ui, uj)
    agreement2 = belief_agreement(vi, vj)
    return belief_discordance(agreement1, agreement2)
end

#=
Degree of Conflict
See equation 18-23
We rename κ to δ_κ since it is acting as our distance

From equation 19
C = [[1, 1, 1, 1],
     [1, 0, 1, 0],
     [1, 1, 0, 0],
     [1, 0, 0, 0]]

δ_κ(m_ij, m'_ij) = m_ij^T * C * m'_ij

But since we are restricting to fuzzy, m_ij = [0, ui ⋅ uj, 1 - ui ⋅ uj, 0]
δ_κ(m_ij, m'_ij) = [0, ui ⋅ uj, 1 - ui ⋅ uj, 0]^T ⋅ C ⋅ [0, vi ⋅ vj, 1 - vi ⋅ vj, 0]
               = [0, ui ⋅ uj, 1 - ui ⋅ uj, 0] ⋅ [1, 1 - vi ⋅ vj, vi ⋅ vj, 0]
               = ui ⋅ uj (1 - vi ⋅ vj) + vi ⋅ vj (1 - ui ⋅ uj)
               = ui ⋅ uj + vi ⋅ vj - 2 * ui⋅uj * vi ⋅ vj

So writing as agreement and discordance
α(i,j) = ui ⋅ uj
disc(i,j) = α^A(i,j) + α^B(i,j) - 2 * α^A(i,j) * α^B(i,j)
=#

function consistency_agreement(ui::Vector{<:Real}, uj::Vector{<:Real}) <: Real
    return dot(ui, uj)
end

function consistency_discordance(agreement1<:Real, agreement2<:Real) <: Real
    return agreement1 + agreement2 - 2 * agreement1 * agreement2
end

function consistency_discordance(ui::Vector{<:Real}, uj::Vector{<:Real}, vi::Vector{<:Real}, vj::Vector{<:Real}) <: Real
    agreement1 = consistency_agreement(ui, uj)
    agreement2 = consistency_agreement(vi, vj)
    return consistency_discordance(agreement1, agreement2)
end