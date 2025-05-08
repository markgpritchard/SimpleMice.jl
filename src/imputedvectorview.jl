# Each `AbstractImputedVectorView` type accessed a vector that combines original non-missing
# values and one of the imputed values for each missing value

abstract type AbstractImputedVectorView{S} <: AbstractVector{S} end

"""
    ImputedVectorView{S, T}

A view of an `ImputedVector`. Allows functions that take an `AbstractVector` to access all
    originally non-missing data and one of the imputed datasets.

Values in the `ImputedVectorView` will be of type `S`, regardless of whether they were 
    originally non-missing or were imputed. 

# Fields
- `imputedvector::T`: stores the array being wrapped
- `index::Int`: stores metadata about the array

# Examples
```
julia> using StableRNGs

julia> rng = StableRNG(1);

julia> v = imputedvector(rng, 5, [ 1, missing, 5 ])
3-element ImputedVector{5, Int64, Float64}:
 1
  [5.0, 1.0, 1.0, 5.0, 5.0]
 5

julia> ImputedVectorView(v, 1)
3-element ImputedVectorView{Float64, ImputedVector{5, Int64, Float64}}:
 1.0
 5.0
 5.0

julia> ImputedVectorView(v, 2)
3-element ImputedVectorView{Float64, ImputedVector{5, Int64, Float64}}:
 1.0
 1.0
 5.0
```
"""
@auto_hash_equals struct ImputedVectorView{S, T} <: AbstractImputedVectorView{S}
    imputedvector::T
    index::Int

    function ImputedVectorView(
        imputedvector::AbstractImputedVector{Ni, U, V}, index
    ) where {Ni, U, V}
        0 < index <= Ni || throw(_imputedtableviewerror(index, Ni))
        combinedtype = typeof(one(U) + one(V))
        return new{combinedtype, typeof(imputedvector)}(imputedvector, index)
    end
end

"""
    imputedvectorview(imputedvector::AbstractImputedVector, index::Int)
    imputedvectorview(v::AbstractVector{<:Union{Missing, <:Number}}, ::Int)

Generate an `ImputedVectorView`. 

If an `AbstractVector{<:Union{Missing, <:Number}}` is passed to the function it returns the 
    original vector. 

# Examples
```
julia> using StableRNGs

julia> rng = StableRNG(1);

julia> v = [ 1, missing, 5 ];

julia> iv = imputedvector(rng, 5, v)
3-element ImputedVector{5, Int64, Float64}:
 1
  [5.0, 1.0, 1.0, 5.0, 5.0]
 5

julia> imputedvectorview(v, 1)
3-element Vector{Union{Missing, Int64}}:
 1
  missing
 5

julia> imputedvectorview(iv, 1)
3-element ImputedVectorView{Float64, ImputedVector{5, Int64, Float64}}:
 1.0
 5.0
 5.0
```
"""
function imputedvectorview(imputedvector::AbstractImputedVector, index::Int)
    return ImputedVectorView(imputedvector, index)
end 

imputedvectorview(v::AbstractVector{<:Number}, ::Int) = v
imputedvectorview(v::AbstractVector{<:Union{Missing, <:Number}}, ::Int) = v

Base.iterate(v::AbstractImputedVectorView) = iterate(v, 1)

function Base.iterate(v::AbstractImputedVectorView, i::Integer) 
    x = getindex(v, i)
    if i == length(v) 
        i_n = nothing 
    else 
        i_n = i + 1
    end 
    return ( x, i_n )
end 

Base.iterate(::AbstractImputedVectorView, ::Nothing) = nothing

function Base.getindex(v::ImputedVectorView{S, T}, i::Integer) where {S, T}
    if isimputedvalue(v, i)
        j = findfirst(x -> x == i, v.imputedvector.missingindex)
        return S(getindex(v.imputedvector, i)[v.index])
    else
        return S(getindex(v.imputedvector, i))
    end
end

Base.length(v::AbstractImputedVectorView) = length(v.imputedvector)
Base.size(v::AbstractImputedVectorView) = size(v.imputedvector)
