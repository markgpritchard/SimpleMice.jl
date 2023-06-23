
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

function returnvalues(n, imputedvalues::Vector{Vector{Int}}) 
    return ImputedVectorInt(n, imputedvalues)
end 

function returnvalues(n, imputedvalues::Vector{Vector{Float64}}) 
    return ImputedVectorFloat64(n, imputedvalues)
end 

function returnvalues(n, imputedvalues::Vector{Vector{Bool}}) 
    return ImputedVectorBool(n, imputedvalues)
end 

function returnvalues(n, imputedvalues::Vector{Vector{<:AbstractString}}) 
    return ImputedVectorString(n, imputedvalues)
end 

function returnvalues(n, imputedvalues::Vector{Vector{T}}) where T <: Any 
    return ImputedVectorAny(n, imputedvalues)
end 
