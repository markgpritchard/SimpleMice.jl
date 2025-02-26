# Each `AbstractImputedVectorView` type accessed a vector that combines original non-missing
# values and one of the imputed values for each missing value

abstract type AbstractImputedVectorView{S} <: AbstractVector{S} end

@auto_hash_equals struct ImputedVectorView{S, T} <: AbstractImputedVectorView{S}
    imputedvector               :: T
    index                       :: Int
end

function ImputedVectorView(
    imputedvector::AbstractImputedVector{Ni, S, T}, index::Int
) where {Ni, S, T}
    @assert index <= Ni "Index, $index, must be no more than Ni, $Ni"
    @assert index > 0 "Index must be positive"
    combinedtype = typeof(one(S) + one(T))
    return ImputedVectorView{combinedtype, typeof(imputedvector)}(imputedvector, index)
end

function imputedvectorview(imputedvector::AbstractImputedVector, index::Int)
    return ImputedVectorView(imputedvector, index)
end 

imputedvectorview(imputedvector::AbstractVector{<:Number}, ::Int) = imputedvector

function imputedvectorview(imputedvector::AbstractVector{<:Union{Missing, <:Number}}, ::Int)
    return imputedvector
end

iterate(v::AbstractImputedVectorView) = iterate(v, 1)

function iterate(v::AbstractImputedVectorView, i::Integer) 
    x = getindex(v, i)
    if i == length(v) 
        i_n = nothing 
    else 
        i_n = i + 1
    end 
    return ( x, i_n )
end 

iterate(::AbstractImputedVectorView, ::Nothing) = nothing

function getindex(v::ImputedVectorView{S, T}, i::Integer) where {S, T}
    if i ∈ v.imputedvector.missingindex
        j = findfirst(x -> x == i, v.imputedvector.missingindex)
        return S(v.imputedvector.imputedvalues[j, v.index])
    else
        return S(v.imputedvector.original[i])
    end
end

length(v::AbstractImputedVectorView) = length(v.imputedvector)
size(v::AbstractImputedVectorView) = size(v.imputedvector)
