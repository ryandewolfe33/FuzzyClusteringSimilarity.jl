function massageMatrix(z::AbstractMatrix{<:Real})
    # Make Hard (Bool values) if possible
    try
        z = convert(Matrix{Bool}, z)
    catch InexactError
    end
    # Make matrices so that each column is a point. Assume #points > #clusters
    if (size(z, 1) > size(z, 2))
        z = transpose(z)
    end
    return copy(z)
end