using FuzzyClusteringSimilarity
using Test

# Tolerance in tests is much higher than expected tolerance. Fixing a random number would require fixing the integration algorithm.
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
        in = [[0.33, 0.33, 0.34] [0.33, 0.33, 0.34]]
        out = massageMatrix(in)
        @test out isa Matrix{Float64}
        @test size(out, 1) == 2
        @test size(out, 2) == 3
    end

    @testset "ndc" begin
    # Correct answer taken from https://github.com/its-likeli-jeff/FARI, assumed correct
        a = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
        b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
        result = ndc(a, b)
        correct = 1
        @test result ≈ correct

        a = [[0.34, 0.33, 0.33] [0.5, 0.5, 0] [0, 0.5, 0.5]]
        b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
        result = ndc(a, b)
        correct = 0.61
        @test result ≈ correct

        a = [[1, 0] [0, 1] [0.5, 0.5]]
        b = [[0.5, 0.5] [0.3, 0.7] [0.5, 0.5]]
        result = ndc(a, b)
        correct = 0.4666666666666
        @test result ≈ correct

        result = ndc(b, a)
        @test result ≈ correct
    end

    @testset "Expectation Fit" begin
        @testset "Two Sided" begin
            a = [[0.34, 0.33, 0.33] [0.5, 0.5, 0] [0, 0.5, 0.5]]
            b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
            result = endc(a, b, "fit")
            @test isapprox(result, 0.499, atol=tol)

            a = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
            result = endc(a, b, "fit")
            @test isapprox(result, 0.501, atol=tol)
            
            a = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
            b = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            result = endc(a, b, "fit")
            @test isapprox(result, 0.500, atol=tol)

            a = [[1, 0] [1, 0] [0, 1]]
            b = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            result = endc(a, b, "fit")
            @test isapprox(result, 0.481, atol=tol)
        end

        @testset "One Sided" begin
            a = [[0.34, 0.33, 0.33] [0.5, 0.5, 0] [0, 0.5, 0.5]]
            b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
            result = endc(a, b, "fit", oneSided=true)
            @test isapprox(result, 0.345, atol=tol)

            a = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
            result = endc(a, b, "fit", oneSided=true)
            @test isapprox(result, 0.34, atol=tol)
            
            a = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
            b = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            result = endc(a, b, "fit", oneSided=true)
            @test isapprox(result, 0.5, atol=tol)

            a = [[1, 0] [1, 0] [0, 1]]
            b = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            result = endc(a, b, "fit", oneSided=true)
            @test isapprox(result, 0.481, atol=tol)
        end
    end

    @testset "Expectation Sym" begin
        @testset "Two Sided" begin
            a = [[0.34, 0.33, 0.33] [0.5, 0.5, 0] [0, 0.5, 0.5]]
            b = [[0.4, 0.6] [0.6, 0.4] [0.5, 0.5]]
            result = endc(a, b, "sym")
            @test isapprox(result, 0.36, atol=tol)

            a = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            b = [[0.4, 0.6] [0.4, 0.6] [0.5, 0.5]]
            result = endc(a, b, "sym")
            @test isapprox(result, 0.36, atol=tol)
            
            a = [[0.4, 0.6] [0.4, 0.6] [0.5, 0.5]]
            b = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            result = endc(a, b, "sym")
            @test isapprox(result, 0.36, atol=tol)

            a = [[1, 0] [1, 0] [0, 1]]
            b = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            result = endc(a, b, "sym")
            @test isapprox(result, 0.5, atol=tol)
        end

        @testset "One Sided" begin
            a = [[0.34, 0.33, 0.33] [0.5, 0.5, 0] [0, 0.5, 0.5]]
            b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
            result = endc(a, b, "sym", oneSided=true)
            @test isapprox(result, 0.33, atol=tol)

            a = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
            result = endc(a, b, "sym", oneSided=true)
            @test isapprox(result, 0.33, atol=tol)
            
            a = [[0.5, 0.5] [0.4, 0.6] [0.4, 0.6]]
            b = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            result = endc(a, b, "sym", oneSided=true)
            @test isapprox(result, 0.09, atol=tol)

            a = [[1, 0] [1, 0] [0, 1]]
            b = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            result = endc(a, b, "sym", oneSided=true)
            @test isapprox(result, 0.5, atol=tol)
        end
    end

    @testset "Expectation Flat" begin
        @testset "Two Sided" begin
            a = [[0.34, 0.33, 0.33] [0.5, 0.5, 0] [0, 0.5, 0.5]]
            b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
            result = endc(a, b, "flat")
            @test isapprox(result, 0.74, atol=tol)

            a = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
            result = endc(a, b, "flat")
            @test isapprox(result, 0.74, atol=tol)
            
            a = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
            b = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            result = endc(a, b, "flat")
            @test isapprox(result, 0.74, atol=tol)

            a = [[1, 0] [1, 0] [0, 1]]
            b = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            result = endc(a, b, "flat")
            @test isapprox(result, 0.74, atol=tol)
        end

        @testset "One Sided" begin
            a = [[0.34, 0.33, 0.33] [0.5, 0.5, 0] [0, 0.5, 0.5]]
            b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
            result = endc(a, b, "flat", oneSided=true)
            @test isapprox(result, 0.6, atol=tol)

            a = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
            result = endc(a, b, "flat", oneSided=true)
            @test isapprox(result, 0.6, atol=tol)
            
            a = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
            b = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            result = endc(a, b, "flat", oneSided=true)
            @test isapprox(result, 0.33, atol=tol)

            a = [[1, 0] [1, 0] [0, 1]]
            b = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
            result = endc(a, b, "flat", oneSided=true)
            @test isapprox(result, 0.33, atol=tol)
        end
    end

    @testset "Permutation" begin
    # Permutation is the same for one or two sided
        a = [[0.34, 0.33, 0.33] [0.5, 0.5, 0] [0, 0.5, 0.5]]
        b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
        result = endc(a, b, "perm", oneSided=true)
        @test isapprox(result, 0.61, atol=tol)

        a = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
        b = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
        result = endc(a, b, "perm", oneSided=true)
        @test isapprox(result, 0.0, atol=tol)
        
        a = [[0.5, 0.5] [0.5, 0.5] [0.5, 0.5]]
        b = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
        result = endc(a, b, "perm", oneSided=true)
        @test isapprox(result, 0.0, atol=tol)

        a = [[1, 0] [1, 0] [0, 1]]
        b = [[1, 0, 0] [0, 1, 0] [0, 0, 1]]
        result = endc(a, b, "perm", oneSided=true)
        @test isapprox(result, 0.67, atol=tol)
    end

    @testset "ANDC" begin
        a = [[0.34, 0.33, 0.33] [0.5, 0.5, 0] [0, 0.5, 0.5]]
        b = [[0.4, 0.6] [0.4, 0.6] [0.5, 0.5]]
        result = andc(a, b, "fit")
        @test isapprox(result, 0.49, atol=tol)
        result = andc(a, b, "fit", oneSided=true)
        @test isapprox(result, 0.5, atol=tol)
        result = andc(a, b, "sym")
        @test isapprox(result, 0.49, atol=tol)
        result = andc(a, b, "sym", oneSided=true)
        @test isapprox(result, 0.51, atol=tol)
        result = andc(a, b, "flat")
        @test isapprox(result, -0.26, atol=tol)
        result = andc(a, b, "flat", oneSided=true)
        @test isapprox(result, 0.03, atol=tol)
        result = andc(a, b, "perm")
        @test isapprox(result, 0.0, atol=tol)
    end
end
