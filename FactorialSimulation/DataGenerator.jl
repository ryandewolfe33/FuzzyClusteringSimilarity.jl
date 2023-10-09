using Distributions
using FuzzyClusteringSimilarity
using JLD
using ProgressMeter
using Base.Threads

function runSimulation()
    clusterNumbers = [2,4,8,16,32,64,128]
    observationNumbers = [128, 256, 512, 1024, 2048, 4096, 8192]
    clusterSizeImbalances = [0.8, 0.6, 0.4, 0.2]
    precisions = [0, 0.01, 0.1,  1, 1.5]
    randomizeRates = [0.2, 0.4, 0.6, 0.8, 1.0]
    nruns = 5

    settingsList = vec([
        [clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex]
        for clusterNumIndex in axes(clusterNumbers, 1),
            observationNumIndex in axes(observationNumbers, 1),
            clusterSizeImbalanceIndex in axes(clusterSizeImbalances, 1),
            precisionIndex in axes(precisions, 1),
            randomizeIndex in axes(randomizeRates, 1)
            ])
    #=
    # Two Sided Comparisons
    results = Array{Float64}(undef, (
        4,
        length(clusterNumbers),
        length(observationNumbers),
        length(clusterSizeImbalances),
        length(precisions),
        length(randomizeRates),
    ))

    p = Progress(length(settingsList), showspeed=true)

    for runNo in nruns
        @threads for settings in settingsList
            clusterNumIndex = settings[1]
            observationNumIndex = settings[2]
            clusterSizeImbalanceIndex = settings[3]
            precisionIndex = settings[4]
            randomizeIndex = settings[5]
            
            #Initialize Clusters
            nClusters = clusterNumbers[clusterNumIndex]
            nObservations = observationNumbers[observationNumIndex]
            clusterSizeImbalance = clusterSizeImbalances[clusterSizeImbalanceIndex]

            clustering1 = zeros(Float64, (nClusters, nObservations))
            clustering2 = zeros(Float64, (nClusters, nObservations))

            # clusterSizeImbalance % of clusters get 80% of points
            nLargeClusters = floor(Int64, clusterSizeImbalance * nClusters)
            totalLargeClusterObservationCount = 0
            if nLargeClusters > 0
                totalLargeClusterObservationCount = ceil(Int64, 0.8 * nObservations)
            end
            nSmallClusters = nClusters - nLargeClusters
            totalSmallClusterObservationCount = nObservations - totalLargeClusterObservationCount


            for i in 1:totalLargeClusterObservationCount
                clusterNum = i % nLargeClusters + 1
                clustering1[clusterNum, i] = 1
                clustering2[clusterNum, i] = 1
            end
            for i in totalLargeClusterObservationCount + 1:nObservations
                clusterNum = i % nSmallClusters + 1 + nLargeClusters
                clustering1[clusterNum, i] = 1
                clustering2[clusterNum, i] = 1
            end

            #Initialize Dirichlet for random points
            # nLargeClusters get 80% of precision
            precision = precisions[precisionIndex]
            
            if precision == 0
                largePrecisionValue = 0.8 / nLargeClusters
                smallPrecisionValue = 0.2 / nSmallClusters
            
                distParameter = Vector{Float64}(undef, nClusters)
                for i in 1:nLargeClusters
                    distParameter[i] = largePrecisionValue
                end
                for i in nLargeClusters + 1:nClusters
                    distParameter[i] = smallPrecisionValue
                end
                    
                randomMembershipDist = Multinomial(1, distParameter/sum(distParameter))
            else
                largePrecisionValue = precision * 0.8 / nLargeClusters
                smallPrecisionValue = precision * 0.2 / nSmallClusters
            
                distParameter = Vector{Float64}(undef, nClusters)
                for i in 1:nLargeClusters
                    distParameter[i] = largePrecisionValue
                end
                for i in nLargeClusters + 1:nClusters
                    distParameter[i] = smallPrecisionValue
                end
                    
                randomMembershipDist = Dirichlet(distParameter)
            end
                    
            # Randomize Clusterings
            randomizeRate = randomizeRates[randomizeIndex]
            nToRandomize = round(Int64, nObservations*randomizeRate)

            indexesToRandomize = sample(axes(clustering1, 2), nToRandomize, replace=false)
            for i in indexesToRandomize
                randMembership = rand(randomMembershipDist)
                while !all(isfinite, randMembership)
                    randMembership = rand(randomMembershipDist)
                end
                clustering1[:, i] = randMembership

                randMembership = rand(randomMembershipDist)
                while !all(isfinite, randMembership)
                    randMembership = rand(randomMembershipDist)
                end
                clustering2[:, i] = randMembership
            end

            # Calculate ANDCs
            results[1, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex] = andc(clustering1, clustering2, "fit")
            results[2, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex] = andc(clustering1, clustering2, "sym")
            results[3, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex] = andc(clustering1, clustering2, "flat")
            results[4, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex] = andc(clustering1, clustering2, "perm")
    

            # Update Progress Bar
            next!(p)
        end

        filename = "/FactorialSimulation/Data/FactorialSimulationTwoSided" * string(runNo) * ".jld"
        save(filename, "results", results)
        finish!(p)
    end
    =#

    # One Sided Comparisons
    results = Array{Float64}(undef, (
        4,
        length(clusterNumbers),
        length(observationNumbers),
        length(clusterSizeImbalances),
        length(precisions),
        length(randomizeRates),
    ))

    p = Progress(length(settingsList), showspeed=true)
    # Two Sided Comparisons
    for runNo in 1:nruns
        @threads for settings in settingsList
            clusterNumIndex = settings[1]
            observationNumIndex = settings[2]
            clusterSizeImbalanceIndex = settings[3]
            precisionIndex = settings[4]
            randomizeIndex = settings[5]
            
            #Initialize Clusters
            nClusters = clusterNumbers[clusterNumIndex]
            nObservations = observationNumbers[observationNumIndex]
            clusterSizeImbalance = clusterSizeImbalances[clusterSizeImbalanceIndex]

            clustering1 = zeros(Float64, (nClusters, nObservations))
            clustering2 = zeros(Float64, (nClusters, nObservations))

            # clusterSizeImbalance % of clusters get 80% of points
            nLargeClusters = floor(Int64, clusterSizeImbalance * nClusters)
            totalLargeClusterObservationCount = 0
            if nLargeClusters > 0
                totalLargeClusterObservationCount = ceil(Int64, 0.8 * nObservations)
            end
            nSmallClusters = nClusters - nLargeClusters
            totalSmallClusterObservationCount = nObservations - totalLargeClusterObservationCount


            for i in 1:totalLargeClusterObservationCount
                clusterNum = i % nLargeClusters + 1
                clustering1[clusterNum, i] = 1
                clustering2[clusterNum, i] = 1
            end
            for i in totalLargeClusterObservationCount + 1:nObservations
                clusterNum = i % nSmallClusters + 1 + nLargeClusters
                clustering1[clusterNum, i] = 1
                clustering2[clusterNum, i] = 1
            end

            #Initialize Dirichlet for random points
            # nLargeClusters get 80% of precision
            precision = precisions[precisionIndex]
            
            if precision == 0
                largePrecisionValue = 0.8 / nLargeClusters
                smallPrecisionValue = 0.2 / nSmallClusters
            
                distParameter = Vector{Float64}(undef, nClusters)
                for i in 1:nLargeClusters
                    distParameter[i] = largePrecisionValue
                end
                for i in nLargeClusters + 1:nClusters
                    distParameter[i] = smallPrecisionValue
                end
                    
                randomMembershipDist = Multinomial(1, distParameter/sum(distParameter))
            else
                largePrecisionValue = precision * 0.8 / nLargeClusters
                smallPrecisionValue = precision * 0.2 / nSmallClusters
            
                distParameter = Vector{Float64}(undef, nClusters)
                for i in 1:nLargeClusters
                    distParameter[i] = largePrecisionValue
                end
                for i in nLargeClusters + 1:nClusters
                    distParameter[i] = smallPrecisionValue
                end
                    
                randomMembershipDist = Dirichlet(distParameter)
            end
                    
            # Randomize Clusterings
            randomizeRate = randomizeRates[randomizeIndex]
            nToRandomize = round(Int64, nObservations*randomizeRate)

            indexesToRandomize = sample(axes(clustering1, 2), nToRandomize, replace=false)
            for i in indexesToRandomize
                randMembership = rand(randomMembershipDist)
                while !all(isfinite, randMembership)
                    randMembership = rand(randomMembershipDist)
                end
                clustering1[:, i] = randMembership
            end

            # Calculate ANDCs
            results[1, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex] = andc(clustering1, clustering2, "fit", oneSided=true)
            results[2, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex] = andc(clustering1, clustering2, "sym", oneSided=true)
            results[3, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex] = andc(clustering1, clustering2, "flat", oneSided=true)
            results[4, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex] = andc(clustering1, clustering2, "perm", oneSided=true)
        
            # Update Progress Bar
            next!(p)
        end

        filename = "FactorialSimulation/Data/FactorialSimulationOneSided" * string(runNo) * ".jld"
        save(filename, "results", results)
        finish!(p)
    end
end

runSimulation()