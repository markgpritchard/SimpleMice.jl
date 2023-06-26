
"""
    getvalues(value::ImputedDataFrame, name::Symbol)

Extracts values from imputed datasets.

Works in the same way as `getproperty` does for a DataFrame. Results are provided 
    in an `ImputedVector`
"""
function getvalues(value::ImputedDataFrame, name)
    n = value.numberimputed 
    originalvector = getproperty(value.originaldf, name)
    imputedvalues = [ getproperty(value.imputeddfs[i], name) for i ∈ 1:n ]
    return returnvalues(originalvector, n, imputedvalues)
end

function returnvalues(originalvector, n, imputedvalues::Vector{Vector{T}}) where T
    return ImputedVector{T}(originalvector, n, imputedvalues)
end 

# No documentation added for this function as already fully described in ?eltype(a)
eltype(a::ImputedVector{T}) where T = T#eltype(T) 
