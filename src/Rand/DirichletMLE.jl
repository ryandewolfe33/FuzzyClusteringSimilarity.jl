using SpecialFunctions


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