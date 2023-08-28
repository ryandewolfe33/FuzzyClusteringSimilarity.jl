using Distributions
using FuzzyClusteringSimilarity
using JLD
using ProgressMeter
using Base.Threads

function runSimulation()
    clusterNumbers = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    observationNumbers = [100, 1000, 5000, 10000]
    clusterSizeImbalances = [0.8, 0.5, 0.2]
    precisions = [0, 0.1, 1, 1.5]
    randomizeRates = [0.2, 0.6, 1.0]
    nruns = 5

    settingsList = vec([
        [clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex, runNo]
        for clusterNumIndex in axes(clusterNumbers, 1),
            observationNumIndex in axes(observationNumbers, 1),
            clusterSizeImbalanceIndex in axes(clusterSizeImbalances, 1),
            precisionIndex in axes(precisions, 1),
            randomizeIndex in axes(randomizeRates, 1),
            runNo in 1:nruns
            ])

    # Two Sided Comparisons
    results = Array{Float64}(undef, (
        4,
        length(clusterNumbers),
        length(observationNumbers),
        length(clusterSizeImbalances),
        length(precisions),
        length(randomizeRates),
        nruns
    ))

    p = Progress(length(settingsList), showspeed=true)

    @threads for settings in settingsList
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

        # Calculate ANDCs
        results[1, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex, runNo] = andc(clustering1, clustering2, "fit")
        results[2, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex, runNo] = andc(clustering1, clustering2, "sym")
        results[3, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex, runNo] = andc(clustering1, clustering2, "flat")
        results[4, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex, runNo] = andc(clustering1, clustering2, "perm")
   

        # Update Progress Bar
        next!(p)
    end

    save("FactorialSimulationTwoSided.jld", "results", results)
    finish!(p)

    # One Sided Comparisons
    results = Array{Float64}(undef, (
        4,
        length(clusterNumbers),
        length(observationNumbers),
        length(clusterSizeImbalances),
        length(precisions),
        length(randomizeRates),
        nruns
    ))

    p = Progress(length(settingsList), showspeed=true)
    # Two Sided Comparisons
    @threads for settings in settingsList
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
        end

        # Calculate ANDCs
        results[1, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex, runNo] = andc(clustering1, clustering2, "fit", oneSided=true)
        results[2, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex, runNo] = andc(clustering1, clustering2, "sym", oneSided=true)
        results[3, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex, runNo] = andc(clustering1, clustering2, "flat", oneSided=true)
        results[4, clusterNumIndex, observationNumIndex, clusterSizeImbalanceIndex, precisionIndex, randomizeIndex, runNo] = andc(clustering1, clustering2, "perm", oneSided=true)
    
        # Update Progress Bar
        next!(p)
    end

    save("FactorialSimulationOneSided.jld", "results", results)
    finish!(p)
end

runSimulation()