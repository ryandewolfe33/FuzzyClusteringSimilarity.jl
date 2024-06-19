struct Permutation <: AbstractRandAdjustment end

function expectedsimilarity(
        z1::AbstractMatrix, z2::AbstractMatrix, index::AbstractAgreementConcordanceIndex, model::Permutation)
    npoints = size(z1, 2)
    n = npoints * (npoints - 1) / 2

    agreements1 = agreements(z1, index)
    agreements2 = agreements(z2, index)

    sort!(agreements1)
    sort!(agreements2)

    return 1 - sum(makeS(agreements1, agreements2)) / n^2
end

# Algorithm 1 from FuzzyAgreement
# https://doi.org/10.1007/s00357-021-09407-3
function makeS(x::AbstractVector, y::AbstractVector)
    n = length(x)
    R = 1
    s = zeros(Float64, n)
    for i in 1:n
        if x[i] < y[1]
            R += 1
            s[1] += y[1] - x[i]
        else
            s[1] += x[i] - y[1]
        end
    end

    for k in 2:n
        L = R
        adjustment = 0
        while R ≤ n && y[k] ≥ x[R]
            adjustment += y[k] - x[R]
            R += 1
        end
        s[k] = s[k - 1] + (2 * L - 2 - n) * (y[k] - y[k - 1]) + 2 * adjustment
    end
    return s
end

function expectedsimilarity(
        z1::AbstractMatrix, z2::AbstractMatrix, index::Frobenious, model::Permutation)
    B1 = transpose(z1) * z1
    B2 = transpose(z2) * z2
    one = ones(size(B1))
    n = size(B1, 2)
    M = one / n
    R = M - I

    term1 = 2 * frobinnerproduct(B1, one) * frobinnerproduct(B2, one) /
            (frobinnerproduct(B1, B1) * frobinnerproduct(B2, B2))
    term2 = frobinnerproduct(M, B1) * frobinnerproduct(M, B2) +
            frobinnerproduct(R, B1) * frobinnerproduct(R, B2) / (n - 1)
    term3 = frobinnerproduct(B1, one)^2 / frobinnerproduct(B1, B1)
    term4 = frobinnerproduct(B2, one)^2 / frobinnerproduct(B2, B2)
    term5 = n^2 - n
    return (term1 * term2 - term3 - term4 + term5) / (n * (n - 1))
end