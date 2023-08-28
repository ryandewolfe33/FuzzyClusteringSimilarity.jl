using Distributions
using FuzzyClusteringSimilarity
using JLD
using ProgressMeter
using Base.Threads

function runSimulation()
    clusterNumbers = [2, 50]
    observationNumbers = [100, 1000]
    clusterSizeImbalances = [0.8, 0.3]
    precisions = [0.1, 1, 1.5]
    randomizeRates = [0.5, 1.0]
    nruns = 10
    nCalcs = 100
    
    settingsList = vec([
            [clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex, runNo]
            for clusterNumIndex in axes(clusterNumbers, 1),
                observationNumIndex in axes(observationNumbers, 1),
                clusterSizeImbalanceIndex in axes(clusterSizeImbalances, 1),
                precisionIndex in axes(precisions, 1),
                randomizeIndex in axes(randomizeRates, 1),
                runNo in 1:nruns
                ])

    results = Array{Float64}(undef, (
        3,
        length(settingsList),
        nCalcs
    ))

    p = Progress(length(settingsList), showspeed=true)

    @threads for i in axes(settingsList, 1)
        settings = settingsList[i]
        clusterNumIndex = settings[1]
        observationNumIndex = settings[2]
        clusterSizeImbalanceIndex = settings[3]
        precisionIndex = settings[4]
        randomizeIndex = settings[5]
        runNo = settings[6]

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
            clustering1[:, i] = rand(randomMembershipDist)
            clustering2[:, i] = rand(randomMembershipDist)
        end

        # Repeat ANDC calculation and store results
        results[1, i, :] = [andc(clustering1, clustering2, "fit") for _ in 1:nCalcs]
        results[2, i, :] = [andc(clustering1, clustering2, "sym") for _ in 1:nCalcs]
        results[3, i, :] = [andc(clustering1, clustering2, "flat") for _ in 1:nCalcs]

        # Update Progress Bar
        next!(p)
    end

    finish!(p)

    save("ErrorAnalysis.jld", "results", results)
end

runSimulation()