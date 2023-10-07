using Distributions
using LinearAlgebra


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
        while !all(isfinite, x1)
            x1 = rand(d)
        end
        x2 = rand(d)
        while !all(isfinite, x2)
            x2 = rand(d)
        end
        xDisagreement = disagreement(x1, x2, q)
        
        #z2 agree (0) with prob pz2agree
        totalDiscorandce += discordance(xDisagreement, 0, p) * pz2disagree
        #z2 disagree(1) with prob 1-pz2agree
        totalDiscorandce += discordance(xDisagreement, 1, p) * (1-pz2disagree)
    end
    return 1 - totalDiscorandce / num
end


function endc(d::Multinomial, z2::AbstractMatrix{<:AbstractFloat}, p::Integer, q::Integer)
    disagreements = makeApproxDisagreements(z2, q)
    probabilityDistAgrees = sum( (d.p).^2 )
    
    totalDiscordance = 0.0
    for i in axes(disagreements,1)
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
        # α is close to hard, or Nan (caused by close to hard mle)
        if all(αi < 1e-4 for αi in α) || !all(isfinite, α)
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
end
   

function fitHard(z::AbstractMatrix{<:AbstractFloat})
        p = vec(sum(z, dims=2))/size(z, 2)
        if !all(isfinite, p)
            print(z)
            print(p)
        end
        return Multinomial(1, p)
end


function symDist(z::AbstractMatrix{<:Integer})
    return Multinomial(1, size(z, 1))
end
 

function symDist(z::AbstractMatrix{<:AbstractFloat})
    precision = mlePrecisionFixedPoint(z)[1]
    numDimensions = size(z, 1)
    if precision / numDimensions < 1e-4 || !all(isfinite, precision)
        return Multinomial(1, size(z, 1))
    end
    return Dirichlet(numDimensions, precision/numDimensions)
end
  

function flatDist(z::AbstractMatrix{<:Real})
    return Dirichlet(ones(size(z, 1)))
end
  

#function logβ(α::T where T<:AbstractFloat, dim::Int64)
#    return Dirichlet(dim, α).lmnB
#end
  

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
  

function approxDisagreementDistribution(z::Matrix, q::Integer, nbins::Integer=1000)
    disagreements = makeDisagreements(z, q)
    pdf = zeros(Float64, nbins)
    for d in disagreements
        bin = min(floor(Int, nbins * d) + 1, nbins)
        pdf[bin] += 1
    end
    pdf /= length(disagreements)
    return pdf
end


function symTwoSided(clustering1::AbstractMatrix, clustering2::AbstractMatrix, p::Integer, q::Integer)
    distribution1 = symDist(clustering1)
    distribution2 = symDist(clustering2)
    return endc(distribution1, distribution2, p, q)
end   


function symTwoSided(clustering1::AbstractMatrix{<:Integer}, clustering2::AbstractMatrix{<:Integer}, p::Integer, q::Integer)
    numClusters1 = size(clustering1, 1)
    numClusters2 = size(clustering2, 1)
    return 1/(numClusters1*numClusters2) + (1-(1/numClusters1))*(1-(1/numClusters2))
end


function symOneSided(clustering1::AbstractMatrix, clustering2::AbstractMatrix, p::Integer, q::Integer)
    distribution1 = symDist(clustering1)
    return endc(distribution1, clustering2, p, q)
end


function fitTwoSided(clustering1::AbstractMatrix, clustering2::AbstractMatrix, p::Integer, q::Integer)
    distribution1 = fitDist(clustering1)
    distribution2 = fitDist(clustering2)
    return endc(distribution1, distribution2, p, q)
end


function fitTwoSided(clustering1::AbstractMatrix{<:Integer}, clustering2::AbstractMatrix{<:Integer}, p::Integer, q::Integer)
    α = vec(sum(clustering1, dims=2))/size(clustering1, 2)
    β = vec(sum(clustering2, dims=2))/size(clustering2, 2)
    
    clustering1agree = sum(α.^2)
    clustering2agree = sum(β.^2)

    f(x) = x*(1-x)
    clustering1disagree = sum(f.(α))
    clustering2disagree = sum(f.(β))
    
    return clustering1agree*clustering2agree + clustering1disagree * clustering2disagree
end


function fitOneSided(clustering1::AbstractMatrix, clustering2::AbstractMatrix, p::Integer, q::Integer)
    distribution1 = fitDist(clustering1)
    return endc(distribution1, clustering2, p, q)
end


function flatTwoSided(clustering1::AbstractMatrix, clustering2::AbstractMatrix, p::Integer, q::Integer)
    distribution1 = flatDist(clustering1)
    distribution2 = flatDist(clustering2)
    return endc(distribution1, distribution2, p, q)
end


function flatOneSided(clustering1::AbstractMatrix, clustering2::AbstractMatrix, p::Integer, q::Integer)
    distribution1 = flatDist(clustering1)
    return endc(distribution1, clustering2, p, q)
end 