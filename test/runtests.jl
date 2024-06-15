using FuzzyClusteringSimilarity
using Test

# Tolerance in tests is much higher than expected tolerance. Fixing a random number would require fixing the integration algorithm.
# TODO fix random number for repeatability
tol = 0.05

@testset "FuzzyClusteringSimilarity.jl" begin
    @testset "Massage Matrix" begin
        # Correct format
        in = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
        out = massageMatrix(in)
        @test out isa Matrix{Float64}
        @test size(out, 1) == 2
        @test size(out, 2) == 3

        # Convert to Int
        in = [[1.0, 0.0] [1.0, 0.0] [1.0, 0.0]]
        out = massageMatrix(in)
        @test out isa Matrix{Bool}

        # Transpose if more rows than columns
        in = [[0.33, 0.33,0.34] [0.33, 0.33,0.34]]
        out = massageMatrix(in)
        @test out isa Matrix{Float64}
        @test size(out, 1) == 2
        @test size(out, 2) == 3
    end

    @testset "ndc" begin
        @test_throws DomainError NDC(-1, 1)
        @test_throws DomainError NDC(1, -1)
        @test_throws DomainError NDC(1, 0)
        @test_throws DomainError NDC(-1, -1)
        @test_throws InexactError NDC(1.5, 1.5)

        ndc = NDC(2, 3)
        @test ndc.p == 2
        @test ndc.q == 3

        ndc = NDC(2, 2.5)
        @test ndc.q == 2.5

        ndc = NDC()
        @test ndc.p == 1
        @test ndc.q == 1

        # Correct answer taken from https://github.com/its-likeli-jeff/FARI, assumed correct
        a = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
        b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
        result = index(a, b, ndc)
        correct = 1
        @test result ≈ correct

        a = [[0.34, 0.33,0.33] [0.5, 0.5,0] [0, 0.5,0.5]]
        b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
        result = index(a, b, ndc)
        correct = 0.61
        @test result ≈ correct

        a = [[1, 0] [0, 1] [0.5, 0.5]]
        b = [[0.5, 0.5] [0.3, 0.7] [0.5, 0.5]]
        result = index(a, b, ndc)
        correct = 0.4666666666666
        @test result ≈ correct

        result = index(b, a, ndc)
        @test result ≈ correct
    end
end