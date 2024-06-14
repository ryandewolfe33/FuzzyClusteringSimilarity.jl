struct FlatDirichlet<:AbstractAgreementConcordance end
struct SymDirichlet<:AbstractAgreementConcordance end
struct FitDirichlet<:AbstractAgreementConcordance end


function expectedindex(
    z1::AbstractMatrix,
    z2::AbstractMatrix,
    model<:AbstractAgreementConcordance,
    index<:AbstractIndex;
    oneSided=True,
    nsamples<:Int=1000000
    ) <:Real

    if oneSided
        dist = fitdist(z2, model)
        return expectation(z1, dist, index)
    else
        dist1 = fitdist(z1, model)
        dist2 = fitdist(z2, model)
        return expectation(dist1, dist2, index)
    end
end


function expectedindex(dist1::Distribution, dist2::Distribution, index<:AbstractIndex, nsamples<:Int=1000000)
    totaldiscordance = 0.0
    for _ in 1:nsamples
        # TODO better method for checking isfinite (nans and infs both appear)
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


function expectedindex(z1::AbstractMatrix{<:AbstractFloat}, dist::Distribution, index<:AbstractIndex, nsamples<:Int=1000000, exact::Bool=false)
    if exact
        return exactexpectedindex(z1::AbstractMatrix{<:AbstractFloat}, dist::Distribution, index<:AbstractIndex, nsamples<:Int=1000000)
    else
        return approxexpectedindex(z1::AbstractMatrix{<:AbstractFloat}, dist::Distribution, index<:AbstractIndex, nsamples<:Int=1000000)
    end
end


function exactexpectedindex(z1::AbstractMatrix{<:AbstractFloat}, dist::Distribution, index<:AbstractIndex, nsamples<:Int=1000000)
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


function approxexpectedindex(z1::AbstractMatrix{<:AbstractFloat}, dist::Distribution, index<:AbstractIndex, nsamples<:Int=1000000, accuracy<:Real=0.001)
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