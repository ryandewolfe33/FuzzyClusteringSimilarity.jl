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

#=
Jousselme's Distance δJ 
See equations 15 in the paper
Define the Jaccard Matrix
J = [[1, 0, 0, 0],
    [0, 1, 0, 1/2],
    [0, 0, 1, 1/2],
    [0, 1/2, 1/2, 1]]

δJ = (0.5 * (m_ij - m'ij)^T * J * (m_ij - m'_ij))^(1/2)

So for clusterings U, V; vectors looks like [0, ui⋅uj, 1-ui⋅uj, 0] and [0, vi⋅vj, 1-vi⋅vj, 0]
where agreement function α is the dot product

Then 
2 * δ_J(vectors) = [0, ui⋅uj - vi⋅vj, vi⋅vj - ui⋅uj, 0]^T * J * [0, ui⋅uj - vi⋅vj, vi⋅vj - ui⋅uj, 0]
                 = [0, ui⋅uj - vi⋅vj, vi⋅vj - ui⋅uj, 0]^T * [0, ui⋅uj - vi⋅vj, vi⋅vj - ui⋅uj, 0]
                 = (ui⋅uj - vi⋅vj)^2 + (vi⋅vj - ui⋅uj)^2
                 = 2(ui⋅uj - vi⋅vj)^2

So writing as agreement and discordance
α(i,j) = ui⋅uj
disc(i,j) = (α^U(i,j) - α^V(i,j))^2
=#
struct Jousseleme <: AbstractAgreementConcordanceIndex end

function agreement(ui::Vector{<:Real}, uj::Vector{<:Real}, index::Jousseleme)::Real
    return dot(ui, uj)
end

function discordance(agreement1 :: Real, agreement2 :: Real, index::Jousseleme)::Real
    return (agreement1 - agreement2)^2
end

#=
Belief Distance δB
See equation 17

δB(vectors) = 1/2 * ( |m(s) - m'(s)| + |m(¬s) - m'(¬s)| + |m(∅) - m'(∅)| )
            = 1/2 * ( |ui⋅uj - vi⋅vj| + |1 - ui⋅uj - 1 + vi⋅vj| )
            = |ui⋅uj - vi⋅vj|

So agreement and discordance are given by 
α(i,j) = ui⋅uj
disc(i,j) = |α^U(i,j) - α^V(i,j)|
=#
struct Belief <: AbstractAgreementConcordanceIndex end

function agreement(ui::Vector{<:Real}, uj::Vector{<:Real}, index::Belief)::Real
    return dot(ui, uj)
end

function discordance(agreement1 :: Real, agreement2 :: Real, index::Belief)::Real
    return abs(agreement1 - agreement2)
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

But since we are restricting to fuzzy, m_ij = [0, ui⋅uj, 1 - ui⋅uj, 0]
δ_κ(m_ij, m'_ij) = [0, ui⋅uj, 1 - ui⋅uj, 0]^T * C * [0, vi⋅vj, 1 - vi⋅vj, 0]
               = [0, ui⋅uj, 1 - ui⋅uj, 0]^T * [1, 1 - vi⋅vj, vi⋅vj, 0]
               = ui⋅uj * (1 - vi⋅vj) + vi⋅vj * (1 - ui⋅uj)
               = ui⋅uj + vi⋅vj - 2 * ui⋅uj * vi⋅vj

So writing as agreement and discordance
α(i,j) = ui⋅uj
disc(i,j) = α^U(i,j) + α^V(i,j) - 2 * α^U(i,j) * α^V(i,j)
=#
struct Consistency <: AbstractAgreementConcordanceIndex end

function agreement(ui::Vector{<:Real}, uj::Vector{<:Real}, index::Consistency)::Real
    return dot(ui, uj)
end

function discordance(agreement1 :: Real, agreement2 :: Real, index::Consistency)::Real
    return agreement1 + agreement2 - 2 * agreement1 * agreement2
end
