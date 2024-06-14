struct FlatDirichlet<:AbstractAgreementConcordance end
struct SymmetricDirichlet<:AbstractAgreementConcordance end
struct FitDirichlet<:AbstractAgreementConcordance end


function expectedindex(
    z1::AbstractMatrix{<:Real},
    z2::AbstractMatrix{<:Real},
    model<:AbstractAgreementConcordance,
    index<:AbstractIndex;
    oneSided=True;
    nsamples<:Int=1000000
    )<:Real

    if oneSided
        dist = fitdist(z2, model)
        return expectedindex(z1, dist, index, nsamples=nsamples)
    else
        dist1 = fitdist(z1, model)
        dist2 = fitdist(z2, model)
        return expectedindex(dist1, dist2, index, nsamples=nsamples)
    end
end


function expectedindex(dist1::Distribution, dist2::Distribution, index<:AbstractIndex; nsamples<:Int=1000000)<:Real
    totaldiscordance = 0.0
    for _ in 1:nsamples
        # TODO better method for checking isfinite (nans and infs both appear)
        # TODO Requesting matrix of random values is probably faster
        x1 = rand(dist1)
        while !all(isfinite, x1)
            x1 = rand(dist1)
        end
        x2 = rand(dist1)
        while !all(isfinite, x2)
            x2 = rand(dist1)
        end
        
        y1 = rand(dist2)
        while !all(isfinite, y1)
            y1 = rand(dist2)
        end
        y2 = rand(dist2)
        while !all(isfinite, y2)
            y2 = rand(dist2)
        end
        
        totaldiscordance += discordance(x1, x2, y1, y2, index)
    end
    return 1 - totaldiscordance/nsamples
end


function expectedindex(z1::AbstractMatrix{<:Real}, dist::Distribution, index<:AbstractIndex; nsamples<:Int=1000000, exact::Bool=false)<:Real
    if exact
        return exactexpectedindex(z1, dist, index, nsamples=nsamples)
    else
        return approxexpectedindex(z1, dist, index, nsamples=nsamples)
    end
end


function exactexpectedindex(z1::AbstractMatrix{<:Real}, dist::Distribution, index<:AbstractIndex; nsamples<:Int=1000000)
    z1agreements = agreement(z1)
    totaldiscordance = 0.0
    for _ in 1:nsamples
        # TODO better method for checking isfinite (nans and infs both appear)
        x1 = rand(dist)
        while !all(isfinite, x1)
            x1 = rand(dist)
        end
        x2 = rand(dist)
        while !all(isfinite, x2)
            x2 = rand(dist)
        end
        xagreement = agreement(x1, x2, index)
        
        #TODO make more clear; new function?
        currenttotaldiscordance = 0.0
        for z1agreement in z1agreements
            currenttotaldiscordance += discordance(z1agreement, xagreement, index)
        end
        totaldiscordance += currenttotaldiscordance/length(z1agreements)
    end
    return 1 - totaldiscordance/nsamples
end


function approxexpectedindex(z1::AbstractMatrix{<:Real}, dist::Distribution, index<:AbstractIndex; nsamples<:Int=1000000, accuracy<:Real=0.001)
    # TODO handle accuracy input, currently hardcoded to 0.001
    nbins = 1000
    z1agreements = agreement(z1, index)
    weights = zeros(Int, nbins + 1)
    for i in z1agreements
        bin = floor(Int, i * accuracy)
        weights[bin] += 1
    end

    totaldiscordance = 0.0
    for _ in 1:nsamples
        x1 = rand(dist)
        while !all(isfinite, x1)
            x1 = rand(dist)
        end
        x2 = rand(dist)
        while !all(isfinite, x2)
            x2 = rand(dist)
        end
        xagreement = agreement(x1, x2, index)

        # TODO clean up or comment that agreement is binnum
        currenttotaldiscordance = 0.0
        for binnum in 1:length(weights)
            z1agreement = (binnum - 1) / nbins
            currenttotaldiscordance += discordance(z1agreement, xagreement, index)*weights(binnum)
        end
        totaldiscordance += currenttotaldiscordance / length(z1agreements)
    end

    return 1 - totaldiscordance / nsamples
end


# Fitting Dirichlet Distributions according to the different DirichletModels
function fitdist(z::AbstractMatrix{<:Real}, model::FlatDirichlet)
    return Dirichlet(ones(size(z, 1)))
end


function fitdist(z::AbstractMatrix{<:Real}, model::SymmetricDirichlet; minprecision<:Real=1e-4)
    precision, error = mlePrecisionFixedPoint(z)
    numDimensions = size(z, 1)
    # If precision is too low approximate with a multinomail
    if precision / numDimensions < minprecision || !all(isfinite, precision)
        return Multinomial(1, size(z, 1))
    end
    return Dirichlet(numDimensions, precision/numDimensions)
end


function fitdist(z::AbstractMatrix{<:Real}, model::FitDirichlet; minprecision<:Real=1e-4)
    try
        α, error = mleFixedPoint(z)
        # α is close to hard, or Nan (caused by close to hard mle) approximate with multinomail
        # TODO could only some of α be nan
        if all(αi < 1e-4 for αi in α) || !all(isfinite, α)
            return fithard(z)
        end
        return Dirichlet(α)
    catch e
        # Domain Error means z is too close to hard
        if isa(e, DomainError)
            return fithard(z)
        else 
            throw(e)
        end
    end
end


function fithard(z::AbstractMatrix{<:Real})
    p = vec(sum(z, dims=2))/size(z, 2)
    return Multinomial(1, p)
end


# Algorithms to find the maximum liklihood of dirichlet Distributions
# The available method in Distributions.jl does not handle very spase well
# TODO Minka citation for equations
function mlePrecisionFixedPoint(points::Matrix{<:AbstractFloat}, tol::AbstractFloat=1e-10, maxIter::Int=convert(Int, 1e5))
    K = size(points, 1)
    centre = 1/K
    m = fill(centre, K)
    return mlePrecisionFixedPoint(points, m, tol, maxIter)
end


function mlePrecisionFixedPoint(points::Matrix{<:AbstractFloat}, m::AbstractVector, tol::AbstractFloat=1e-10, maxIter::Int=convert(Int, 1e5))
    K = size(points, 1)
    logp̄ = vec(mean(log, points, dims=2))
    logp̄ = max.(logp̄, -1e30)
    
    # Initialize via equation 42 from Minka
    precision = ((K-1)/2) / -sum(m .* (logp̄ - log.(m)))
    
    converged = false
    iteration = 0
    Δ = 1.
    while !converged
        precisionOld = precision
        precisionInv = (K-1)/precisionOld - digamma(precisionOld) + sum(m .* digamma.(precisionOld * m)) - sum(m .* logp̄)
        precision = (K-1)/precisionInv
        
        # precision must be positive
        if precision <= 0
            precision = 1e-30
        end
        
        Δ = abs(precision - precisionOld)
        if  Δ < tol || iteration == maxIter
            converged = true
        end
        iteration += 1
    end
    return (precision, Δ)
end


function mleFixedPoint(points::Matrix{<:AbstractFloat}, tol::AbstractFloat=1e-10, maxIter::Int=convert(Int, 1e5))
    # Initialize alpha using moments
    Ep1 = mean(points[1, :])
    Ep12 = mean(points[1, :].^2)
    Σα = (Ep1 - Ep12) / (Ep12 - Ep1^2)
    α = vec(mean(points, dims=2) / Σα)

    #If points is hard, initial precision is 0
    if Σα == 0
        throw(DomainError("Matrix is hard"))
    end
    
    #Pre compute Logp
    logp̄ = vec(mean(log, points, dims=2))
    logp̄ = max.(logp̄, -1e30)
    
    # Fixed Point iterate
    converged = false
    iteration = 0
    Δ = 1.
    while !converged
        αOld = deepcopy(α)
        for k in eachindex(α)
            α[k] = invdigamma( digamma(sum(αOld)) + logp̄[k] )
        end
        
        Δ = sum(abs.(α - αOld))
        if  Δ < tol || iteration == maxIter
            converged = true
        end
        iteration += 1
    end
    return (α, Δ)
end