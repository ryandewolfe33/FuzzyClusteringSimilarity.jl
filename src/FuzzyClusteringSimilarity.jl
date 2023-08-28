module FuzzyClusteringSimilarity

include("Rand/ANDC.jl")
include("Rand/PermutationModel.jl")
include("Rand/OtherModels.jl")
include("Rand/DirichletMLE.jl")
include("Rand/FARI.jl")

include("LossFunctions.jl")

export andc
export ndc
export endc
export discordance
export FARI
export massageMatrix
export MSE
export MAE
export MPE
export Logloss

function massageMatrix(z::AbstractMatrix)
    # Make Hard (Bool values) if possible
    try
        z = convert(Matrix{Bool}, z)
    catch
    end
    # Make matrices so that each column is a point. Assume #points > #clusters
    if(size(z, 1) > size(z, 2))
        z = transpose(z)
    end
    return copy(z)
end

end
