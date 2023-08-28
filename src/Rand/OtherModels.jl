using Distributions
using LinearAlgebra

#TODO Better disagreement distribution approximation


function disagreement(x::Vector{<:Real}, y::Vector{<:Real}, q::Integer=1)
    return norm(x-y, q)/(2^(1/q))
end

function discordance(
    x1::Vector{<:Real},
    x2::Vector{<:Real},
    y1::Vector{<:Real},
    y2::Vector{<:Real},
    p::Integer=1,
    q::Integer=1
    )
    disagreement1 = disagreement(x1, x2, q)
    disagreement2 = disagreement(y1, y2, q)
    return discordance(disagreement1, disagreement2, p)
end

function discordance(disagreement1::Real, disagreement2::Real, p::Integer=1)
    return abs(disagreement1 - disagreement2)^p
end

function endc(d1::Distribution, d2::Distribution, p=1, q=1)
    num = 1000000
    totalDiscordance = 0.0
    
    for _ in 1:num
        x1 = rand(d1)
        while !all(isfinite, x1)
            x1 = rand(d1)
        end
        x2 = rand(d1)
        while !all(isfinite, x2)
            x2 = rand(d2)
        end
        
        y1 = rand(d2)
        while !all(isfinite, y1)
            y1 = rand(d2)
        end
        y2 = rand(d2)
        while !all(isfinite, y2)
            y2 = rand(d2)
        end
        
        totalDiscordance += discordance(x1, x2, y1, y2, p, q)
    end
    return 1 - totalDiscordance/num
end

function endc(d::Distribution, z2::AbstractMatrix{<:AbstractFloat}, p::Integer, q::Integer)
    num = 100000
    z2Disagreements = approxDisagreementDistribution(z2, q)
    totalDiscordance = 0.0
    for _ in 1:num
        x1 = rand(d)
        while !all(isfinite, x1)
            x1 = rand(d)
        end
        x2 = rand(d)
        while !all(isfinite, x2)
            x2 = rand(d)
        end
        xDisagreement = disagreement(x1, x2, q)
        for j in axes(z2Disagreements, 1)
            yDisagreement = j/length(z2Disagreements)
            totalDiscordance += discordance(xDisagreement, yDisagreement, p) * z2Disagreements[j]
        end
    end
    return 1 - totalDiscordance / num
end

function endc(d::Distribution, z2::AbstractMatrix{<:Integer}, p::Integer, q::Integer)
    num = 100000
    
    # Get Concordance Probability from z2
    α = vec(sum(z2, dims=2))/size(z2, 2)
    pz2disagree = sum(x -> x * (x-1/size(z2, 2)), α)
    
    #Generate random samples
    x1 = Float64[]
    x2 = Float64[]
    totalDiscorandce = 0.
    for _ in 1:num
        x1 = rand(d)
        x2 = rand(d)
        xDisagreement = disagreement(x1, x2, q)
        
        #z2 agree (0) with prob pz2agree
        totalDiscorandce += discordance(xDisagreement, 0, p) * pz2disagree
        #z2 disagree(1) with prob 1-pz2agree
        totalDiscorandce += discordance(xDisagreement, 1, p) * (1-pz2disagree)
    end
    return 1 - totalDiscorandce / num
end

function endc(d::Multinomial, z2::AbstractMatrix{<:AbstractFloat}, p::Integer, q::Integer)
    a = makeDisagreements(z2, q)
    probabilityDistAgrees = sum( (d.p).^2 )
    
    totalDiscordance = 0.0
    for i in axes(a,1)
        #dist agree
        totalDiscordance += discordance(0, a[i], p) * probabilityDistAgrees
        #dist disagree
        totalDiscordance += discordance(1, a[i], p) * (1-probabilityDistAgrees)
    end
    return 1 - totalDiscordance / length(a)
end

function endc(d::Multinomial, z2::AbstractMatrix{<:Integer}, p::Integer, q::Integer)
    pDAgree = sum( (d.p).^2)
    zPVector = sum(z2, dims=2)/size(z2, 2)
    pZAgree = sum( zPVector.^2 )
    return pDAgree*pZAgree + (1-pZAgree)*(1-pDAgree)
end

function fitDist(z::AbstractMatrix{<:Integer})
    p = vec(sum(z, dims=2))/size(z, 2)
    return Multinomial(1, p)
end

function fitDist(z::AbstractMatrix{<:AbstractFloat})
    try
        α = mleFixedPoint(z)[1]
        if all(αi < 1e-4 for αi in α)
            return fitHard(z)
        end
        return Dirichlet(α)
    catch e
        # Domain Error means z is too close to hard
        if isa(e, DomainError)
            return fitHard(z)
        else 
            throw(e)
        end
    end
    # Approximate sparse dirichlets as categorical
end
    
function fitHard(z::AbstractMatrix{<:AbstractFloat})
        p = vec(sum(z, dims=2))/size(z, 2)
        return Multinomial(1, p)
end

function symDist(z::AbstractMatrix{<:Integer})
    return Multinomial(1, size(z, 1))
end
    
function symDist(z::AbstractMatrix{<:AbstractFloat})
    α = mlePrecisionFixedPoint(z)[1]
    numDimensions = size(z, 1)
    if α / numDimensions < 1e-4
        return Multinomial(1, size(z, 1))
    end
    return Dirichlet(numDimensions, α/numDimensions)
end
    
function flatDist(z::AbstractMatrix{<:Real})
    return Dirichlet(ones(size(z, 1)))
end
    
function logβ(α::T where T<:AbstractFloat, dim::Int64)
    return Dirichlet(dim, α).lmnB
end
    
function makeDisagreements(z::AbstractMatrix, q::Integer=1)
    npoints = size(z, 2)
    n = npoints * (npoints-1) / 2
    disagreements = Vector{Float64}(undef, convert(Int, n))
            
    index = 1
    for i in 2:npoints
        for j in 1:i-1
            disagreements[index] = disagreement(z[:, i], z[:, j], q)
            index += 1
        end
    end
    return disagreements
end
        
function approxDisagreementDistribution(z::Matrix, q::Integer)
    disagreements = makeDisagreements(z, q)
    pdf = zeros(Float64, 1000)
    for d in disagreements
        bin = min(floor(Int, 1000 * d) + 1, 1000)
        pdf[bin] += 1
    end
    pdf /= length(disagreements)
    return pdf
end

function symTwoSided(z1::AbstractMatrix, z2::AbstractMatrix, p::Integer, q::Integer)
    d1 = symDist(z1)
    d2 = symDist(z2)
    return endc(d1, d2, p, q)
end   

function symTwoSided(z1::AbstractMatrix{<:Integer}, z2::AbstractMatrix{<:Integer}, p::Integer, q::Integer)
    c1 = size(z1, 1)
    c2 = size(z2, 1)
    return 1/(c1*c2) + (1-(1/c1))*(1-(1/c2))
end

function symOneSided(z1::AbstractMatrix, z2::AbstractMatrix, p::Integer, q::Integer)
    d1 = symDist(z1)
    return endc(d1, z2, p, q)
end

function fitTwoSided(z1::AbstractMatrix, z2::AbstractMatrix, p::Integer, q::Integer)
    d1 = fitDist(z1)
    d2 = fitDist(z2)
    return endc(d1, d2, p, q)
end

function fitTwoSided(z1::AbstractMatrix{<:Integer}, z2::AbstractMatrix{<:Integer}, p::Integer, q::Integer)
    α = vec(sum(z1, dims=2))/size(z1, 2)
    β = vec(sum(z2, dims=2))/size(z2, 2)
    
    z1agree = sum(α.^2)
    z2agree = sum(β.^2)

    f(x) = x*(1-x)
    z1disagree = sum(f.(α))
    z2disagree = sum(f.(β))
    
    return z1agree*z2agree + z1disagree * z2disagree
end

function fitOneSided(z1::AbstractMatrix, z2::AbstractMatrix, p::Integer, q::Integer)
    d1 = fitDist(z1)
    return endc(d1, z2, p, q)
end

function flatTwoSided(z1::AbstractMatrix, z2::AbstractMatrix, p::Integer, q::Integer)
    d1 = flatDist(z1)
    d2 = flatDist(z2)
    return endc(d1, d2, p, q)
end

function flatOneSided(z1::AbstractMatrix, z2::AbstractMatrix, p::Integer, q::Integer)
    d1 = flatDist(z1)
    return endc(d1, z2, p, q)
end 