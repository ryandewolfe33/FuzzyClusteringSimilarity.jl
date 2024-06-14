module FuzzyClusteringSimilarity

include("Rand/Index.jl")
include("Rand/Adjustment.jl")

# Indexes
export AbstractIndex
# From DempsterShafer
export Jousseleme
export Belief
export Consistency
# From NDC
export NDC

# Adjustments
export AbstractRandAdjustment
export AbstractAgreementConcordance
# From DirichletModels
export FlatDirichlet
export SymmetricDirichlet
export FitDirichlet
# From Permutation
export Permutation

# Utils
export massageMatrix

end