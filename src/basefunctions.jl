
"""
    getvalues(value::ImputedDataFrame, name::Symbol)

Extracts values from imputed datasets.

Works in the same way as `getproperty` does for a DataFrame. Results are provided 
    in an `ImputedVector`
"""
function getvalues(value::ImputedDataFrame, name::Symbol)
    n = value.numberimputed 
    imputedvalues = [ getproperty(value.imputeddfs[i], name) for i ∈ 1:n ]
    return returnvalues(n, imputedvalues)
end

function returnvalues(n, imputedvalues::Vector{Vector{T}}) where T
    return ImputedVector{T}(n, imputedvalues)
end 
