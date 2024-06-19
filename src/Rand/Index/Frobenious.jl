#= 
Andrews, J.L., Browne, R. & Hvingelby, C.D.
On Assessments of Agreement Between Fuzzy Partitions. J Classif 39, 326–342 (2022).
https://doi.org/10.1007/s00357-021-09407-3
=#

struct Frobenious <: AbstractIndex
end

function similarity(z1::AbstractMatrix{<:Real}, z2::AbstractMatrix{<:Real}, index::Frobenious)
    B1 = transpose(z1) * z1
    B2 = transpose(z2) * z2
    @info size(B1)
    n = size(B1, 2)
    one = ones(size(B1))
    return (frobinnerproduct(N(B1), N(B2)) + frobinnerproduct(one - N(B1), one - N(B2)) -
            n) / (n * n - 1)
end

function N(B)
    one = ones(size(B))
    return (frobinnerproduct(B, B) / frobinnerproduct(B, one)) .* B
end

function frobinnerproduct(A, B)
    return sum(A .* B)
end