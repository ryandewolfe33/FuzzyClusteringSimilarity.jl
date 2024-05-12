import FuzzyClusteringSimilarity.Rand.DempsterShafer
using Test

@testset "Jousselme" begin
    ui = [0.5, 0.5, 0]
    uj = [0, 0.5, 0.5]
    vi = [0.5, 0.5, 0]
    vj = [0, 0.5, 0.5]
    @test jousseleme_agreement(ui, uj) == 0.25
    @test jousseleme_discordance(ui, uj, vi, vj) == 0

    ui = [1, 0, 0]
    uj = [1, 0, 0]
    vi = [0.5, 0.5, 0]
    vj = [0, 0, 1]
    @test jousseleme_agreement(ui, uj) == 1
    @test jousseleme_agreement(vi, vj) == 0
    @test jousseleme_discordance(1, 0) == 2
    @test jousseleme_discordance(ui, uj, vi, vj) == 1
    @test Jousseleme_discordance(uj, ui, vj, vi) == 1
    @test jousseleme_discordance(vi, vj, ui, uj) == 1
    @test jousseleme_discordance(vj, vi, uj, ui) == 1
end
