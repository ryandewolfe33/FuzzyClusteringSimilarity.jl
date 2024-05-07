struct FlatDirichlet<:AbstractAgreementConcordance end
struct SymDirichlet<:AbstractAgreementConcordance end
struct FitDirichlet<:AbstractAgreementConcordance end

function expectation(
    z1::AbstractMatrix,
    z2::AbstractMatrix,
    model<:AbstractAgreementConcordance,
    index<:AbstractIndex;
    oneSided=True
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

function expectation(dist1::Distribution, dist2::Distribution, index<:AbstractIndex)
    nsamples = 1000000
    totaldiscordance = 0.0
    
    for _ in 1:nsamples
        x1 = rand(dist1)
        while !all(isfinite, x1)
            x1 = rand(d1)
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

function expectation(z1::AbstractMatrix{<:AbstractFloat}, dist::Distribution, index<:AbstractIndex)
    nsamples = 100000
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