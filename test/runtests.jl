using FuzzyClusteringSimilarity
using Test

# Tolerance in tests is much higher than expected tolerance. Fixing a random number would require fixing the integration algorithm.
# TODO fix random number for repeatability
tol = 0.05

@testset verbose=true "FuzzyClusteringSimilarity.jl" begin
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
        result = similarity(a, b, ndc)
        correct = 1
        @test result ≈ correct

        a = [[0.34, 0.33,0.33] [0.5, 0.5,0] [0, 0.5,0.5]]
        b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
        result = similarity(a, b, ndc)
        correct = 0.61
        @test result ≈ correct

        a = [[1, 0] [0, 1] [0.5, 0.5]]
        b = [[0.5, 0.5] [0.3, 0.7] [0.5, 0.5]]
        result = similarity(a, b, ndc)
        correct = 0.4666666666666
        @test result ≈ correct

        result = similarity(b, a, ndc)
        @test result ≈ correct
    end

    @testset verbose=true "Dirichlet Models" begin
        # Using NDC for all tests for simplicity
        ndc = NDC()
        @testset "Expectation Fit" begin
            fit = FitDirichlet()
            @testset "Two Sided" begin
                a = [[0.34, 0.33,0.33] [0.5, 0.5,0] [0, 0.5,0.5]]
                b = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                result = expectedsimilarity(a, b, ndc, fit, onesided = false)
                @test isapprox(result, 0.38, atol = tol)

                a = [[1, 0,0] [0, 1,0] [0, 0,1]]
                b = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                result = expectedsimilarity(a, b, ndc, fit, onesided = false)
                @test isapprox(result, 0.36, atol = tol)

                a = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                b = [[1, 0,0] [0, 1,0] [0, 0,1]]
                result = expectedsimilarity(a, b, ndc, fit, onesided = false)
                @test isapprox(result, 0.36, atol = tol)

                a = [[1, 0] [1, 0] [0, 1]]
                b = [[1, 0,0] [0, 1,0] [0, 0,1]]
                result = expectedsimilarity(a, b, ndc, fit, onesided = false)
                @test isapprox(result, 0.50, atol = tol)
            end

            @testset "One Sided" begin
                a = [[0.34, 0.33,0.33] [0.5, 0.5,0] [0, 0.5,0.5]]
                b = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                result = expectedsimilarity(a, b, ndc, fit)
                @test isapprox(result, 0.70, atol = tol)

                a = [[1, 0,0] [0, 1,0] [0, 0,1]]
                b = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                result = expectedsimilarity(a, b, ndc, fit)
                @test isapprox(result, 0.09, atol = tol)

                a = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                b = [[1, 0,0] [0, 1,0] [0, 0,1]]
                result = expectedsimilarity(a, b, ndc, fit)
                @test isapprox(result, 0.34, atol = tol)

                a = [[1, 0] [1, 0] [0, 1]]
                a = [[1, 0] [1, 0] [0, 1]]
                result = expectedsimilarity(a, b, ndc, fit)
                @test isapprox(result, 0.55, atol = tol)
            end
        end

        @testset "Expectation Sym" begin
            sym = SymmetricDirichlet()
            @testset "Two Sided" begin
                a = [[0.34, 0.33,0.33] [0.5, 0.5,0] [0, 0.5,0.5]]
                b = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                result = expectedsimilarity(a, b, ndc, sym, onesided = false)
                @test isapprox(result, 0.36, atol = tol)

                a = [[1, 0,0] [0, 1,0] [0, 0,1]]
                b = [[0.4, 0.6] [0.4, 0.6] [0.5, 0.5]]
                result = expectedsimilarity(a, b, ndc, sym, onesided = false)
                @test isapprox(result, 0.36, atol = tol)

                a = [[0.4, 0.6] [0.4, 0.6] [0.5, 0.5]]
                b = [[1, 0,0] [0, 1,0] [0, 0,1]]
                result = expectedsimilarity(a, b, ndc, sym, onesided = false)
                @test isapprox(result, 0.36, atol = tol)

                a = [[1, 0] [1, 0] [0, 1]]
                b = [[1, 0,0] [0, 1,0] [0, 0,1]]
                result = expectedsimilarity(a, b, ndc, sym, onesided = false)
                @test isapprox(result, 0.55, atol = tol)
            end

            @testset "One Sided" begin
                a = [[0.34, 0.33,0.33] [0.5, 0.5,0] [0, 0.5,0.5]]
                b = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                result = expectedsimilarity(a, b, ndc, sym)
                @test isapprox(result, 0.70, atol = tol)

                a = [[1, 0,0] [0, 1,0] [0, 0,1]]
                b = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                result = expectedsimilarity(a, b, ndc, sym)
                @test isapprox(result, 0.09, atol = tol)

                a = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                b = [[1, 0,0] [0, 1,0] [0, 0,1]]
                result = expectedsimilarity(a, b, ndc, sym)
                @test isapprox(result, 0.37, atol = tol)

                a = [[1, 0] [1, 0] [0, 1]]
                b = [[1, 0,0] [0, 1,0] [0, 0,1]]
                result = expectedsimilarity(a, b, ndc, sym)
                @test isapprox(result, 0.55, atol = tol)
            end
        end

        @testset "Expectation Flat" begin
            flat = FlatDirichlet()
            @testset "Two Sided" begin
                a = [[0.34, 0.33,0.33] [0.5, 0.5,0] [0, 0.5,0.5]]
                b = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                result = expectedsimilarity(a, b, ndc, flat, onesided = false)
                @test isapprox(result, 0.74, atol = tol)

                a = [[1, 0,0] [0, 1,0] [0, 0,1]]
                b = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                result = expectedsimilarity(a, b, ndc, flat, onesided = false)
                @test isapprox(result, 0.65, atol = tol)

                a = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                b = [[1, 0,0] [0, 1,0] [0, 0,1]]
                result = expectedsimilarity(a, b, ndc, flat, onesided = false)
                @test isapprox(result, 0.65, atol = tol)

                a = [[1, 0] [1, 0] [0, 1]]
                b = [[1, 0,0] [0, 1,0] [0, 0,1]]
                result = expectedsimilarity(a, b, ndc, flat, onesided = false)
                @test isapprox(result, 0.67, atol = tol)
            end

            @testset "One Sided" begin
                a = [[0.34, 0.33,0.33] [0.5, 0.5,0] [0, 0.5,0.5]]
                b = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                result = expectedsimilarity(a, b, ndc, flat)
                @test isapprox(result, 0.78, atol = tol)

                a = [[1, 0,0] [0, 1,0] [0, 0,1]]
                b = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                result = expectedsimilarity(a, b, ndc, flat)
                @test isapprox(result, 0.33, atol = tol)

                a = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
                b = [[1, 0,0] [0, 1,0] [0, 0,1]]
                result = expectedsimilarity(a, b, ndc, flat)
                @test isapprox(result, 0.57, atol = tol)

                a = [[1, 0] [1, 0] [0, 1]]
                b = [[1, 0,0] [0, 1,0] [0, 0,1]]
                result = expectedsimilarity(a, b, ndc, flat)
                @test isapprox(result, 0.52, atol = tol)
            end
        end

        @testset "Permutation" begin
            # Permutation is the same for one or two sided
            perm = Permutation()
            a = [[0.34, 0.33,0.33] [0.5, 0.5,0] [0, 0.5,0.5]]
            b = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
            result = expectedsimilarity(a, b, ndc, perm)
            @test isapprox(result, 0.74, atol = tol)

            a = [[1, 0,0] [0, 1,0] [0, 0,1]]
            b = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
            result = expectedsimilarity(a, b, ndc, perm)
            @test isapprox(result, 0.13, atol = tol)

            a = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
            b = [[1, 0,0] [0, 1,0] [0, 0,1]]
            result = expectedsimilarity(a, b, ndc, perm)
            @test isapprox(result, 0.13, atol = tol)

            a = [[1, 0] [1, 0] [0, 1]]
            b = [[1, 0,0] [0, 1,0] [0, 0,1]]
            result = expectedsimilarity(a, b, ndc, perm)
            @test isapprox(result, 0.67, atol = tol)
        end
    end

    @testset "Frobenious" begin
        fri = Frobenious()

        a = [[0.3, 0.7] [0.52, 0.48] [0.5, 0.5]]
        b = [[0.3, 0.7] [0.52, 0.48] [0.5, 0.5]]
        result = similarity(a, b, fri)
        @test result == 1

        a = [[0.34, 0.33,0.33] [0.5, 0.5,0] [0, 0.5,0.5]]
        b = [[0.48, 0.52] [0.52, 0.48] [0.5, 0.5]]
        result = similarity(a, b, fri)
        correct = 0.91
        @test isapprox(result, correct, atol = tol)

        # Taken from paper
        z1 = [[0.278, 0.378,0.344] [0.361, 0.339,0.300] [0.298, 0.325,0.378] [0.319,
            0.319,0.362] [0.316, 0.379,0.304]]
        z2 = [[0.305, 0.327,0.368] [0.334, 0.344,0.323] [0.364, 0.324,0.312] [0.296,
            0.388,0.316] [0.321, 0.342,0.337]]

        result = adjustedsimilarity(z1, z2, fri, Permutation())
        @test result ≈ -0.3056
        # Expected Similarity is 0.9999551958029949
        # Similarity is 0.9999369353636434
        # This test is failing but maybe its just a numberical error? Probably Not
    end

    @testset "DempsterShafer" begin
        j = Jousseleme()
        b = Belief()
        c = Consistency()

        z1 = [[0.3, 0.7] [0.52, 0.48] [0.5, 0.5]]
        z2 = [[0.3, 0.7] [0.52, 0.48] [0.5, 0.5]]
        result = similarity(z1, z2, j)
        @test result == 1
        result = similarity(z1, z2, b)
        @test result == 1
        result = similarity(z1, z2, b)
        @test result == 1

        z1 = [[0.34, 0.33,0.33] [0.5, 0.5,0] [0, 0.5,0.5]]
        z2 = [[0.48, 0.52] [0.52, 0.48] [0.5, 0.5]]
        result = similarity(z1, z2, j)
        @test isapprox(result, 0.96, atol = tol)
        result = similarity(z1, z2, b)
        @test isapprox(result, 0.80, atol = tol)
        result = similarity(z1, z2, c)
        @test isapprox(result, 0.5, atol = tol)

        z1 = [[0.278, 0.378,0.344] [0.361, 0.339,0.300] [0.298, 0.325,0.378] [0.319,
            0.319,0.362] [0.316, 0.379,0.304]]
        z2 = [[0.305, 0.327,0.368] [0.334, 0.344,0.323] [0.364, 0.324,0.312] [0.296,
            0.388,0.316] [0.321, 0.342,0.337]]

        result = adjustedsimilarity(z1, z2, j, FitDirichlet())
        @test isapprox(result, -0.13, atol=tol)
        result = adjustedsimilarity(z1, z2, j, SymmetricDirichlet())
        @test isapprox(result, -0.22, atol=tol)
        result = adjustedsimilarity(z1, z2, j, FlatDirichlet())
        @test isapprox(result, 0.99, atol=tol)
        result = adjustedsimilarity(z1, z2, j, Permutation())
        @test isapprox(result, 0.99, atol=tol)

        result = adjustedsimilarity(z1, z2, b, FitDirichlet())
        @test isapprox(result, -0.05, atol=tol)
        result = adjustedsimilarity(z1, z2, b, SymmetricDirichlet())
        @test isapprox(result, -0.09, atol=tol)
        result = adjustedsimilarity(z1, z2, b, FlatDirichlet())
        @test isapprox(result, 0.98, atol=tol)
        result = adjustedsimilarity(z1, z2, b, Permutation())
        @test isapprox(result, -0.19, atol=tol)

        result = adjustedsimilarity(z1, z2, c, FitDirichlet())
        @test isapprox(result, 0.0, atol=tol)
        result = adjustedsimilarity(z1, z2, c, SymmetricDirichlet())
        @test isapprox(result, 0.0, atol=tol)
        result = adjustedsimilarity(z1, z2, c, FlatDirichlet())
        @test isapprox(result, 0.0, atol=tol)
        result = adjustedsimilarity(z1, z2, c, Permutation())
        @test isapprox(result, -270.52, atol=tol)

    end

    # TODO Test set for DempsterShafer
end