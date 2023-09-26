function permutationModel(z1::AbstractMatrix, z2::AbstractMatrix,  p::Integer=1, q::Integer=1)
    npoints = size(z1, 2)
    n = npoints * (npoints-1)/2
    
    agreements1 = makeAgreements(z1, q)
    agreements2 = makeAgreements(z2, q)
    
    sort!(agreements1)
    sort!(agreements2)
        
    return 1 - sum(makeS(agreements1, agreements2))/n^2
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
        s[k] = s[k-1] + (2*L - 2 - n)*(y[k] - y[k-1]) + 2*adjustment
    end
    return s
end

function makeAgreements(z::AbstractMatrix, q::Integer=1)
    disagreements = makeDisagreements(z, q)
    return ones(Float64, axes(disagreements, 1)) - disagreements
end
