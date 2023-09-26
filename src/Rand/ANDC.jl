function andc(z1::AbstractMatrix, z2::AbstractMatrix, model::String; oneSided::Bool=false,  p::Integer=1, q::Integer=1)
    #Prep inputs
    z1 = massageMatrix(z1)
    z2 = massageMatrix(z2)

    #Make ANDC
    expected = endc(z1, z2, model, oneSided=oneSided, p=p, q=q)
    observed = ndc(z1, z2, p=p, q=q)
    return (observed - expected) / (1 - expected)
end


function ndc(z1::AbstractMatrix, z2::AbstractMatrix; p::Integer=1, q::Integer=1)
    z1Agreements = makeAgreements(z1, q)
    z2Agreements = makeAgreements(z2, q)
    disc = discordance.(z1Agreements, z2Agreements, p)
    return 1 - mean(disc)
end


function endc(z1::AbstractMatrix, z2::AbstractMatrix, model::String; oneSided=false, p::Integer=1, q::Integer=1)
    # Permutation Model is the same for one and two sided
    if model=="perm"
        return permutationModel(z1, z2, p, q)
    end
        
    if oneSided
    # One Sided Models
        if model=="fit"
            return fitOneSided(z1, z2, p, q)
        elseif model=="sym"
            return symOneSided(z1, z2, p, q)
        elseif model=="flat"
            return flatOneSided(z1, z2, p, q)
        end

    else
    # Two Sided Models
        if model=="fit"
            return fitTwoSided(z1, z2, p, q)
        elseif model=="sym"
            return symTwoSided(z1, z2, p, q)
        elseif model=="flat"
            return flatTwoSided(z1, z2, p, q)
        end
    end
    throw(ArgumentError(concat(model, " does not exist.")))
end