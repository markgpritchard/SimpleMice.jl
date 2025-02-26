# Each `AbstractImputedVector` type gives a vector that combines original non-missing values
# (of type `S`) and a series of `Ni` imputed values (of type `T`)

abstract type AbstractImputedVector{Ni, S, T} <: AbstractVector{S} end  # not exported

struct ImputedVectorMStatic{Ni, Nm, Np, S, T} <: AbstractImputedVector{Ni, S, T}  # not exported
    original                    :: Vector{<:Union{Missing, S}} 
    missingindex                :: SVector{Nm, Int}
    imputedvalues               :: MMatrix{Nm, Ni, T, Np} 
end

struct ImputedVectorStatic{Ni, Nm, Np, S, T} <: AbstractImputedVector{Ni, S, T}  # not exported
    original                    :: Vector{<:Union{Missing, S}} 
    missingindex                :: SVector{Nm, Int}
    imputedvalues               :: SMatrix{Nm, Ni, T, Np} 
end

struct ImputedVector{Ni, Nm, Np, S, T} <: AbstractImputedVector{Ni, S, T}  # not exported
    original                    :: Vector{<:Union{Missing, S}}
    missingindex                :: Vector{Int}
    imputedvalues               :: Matrix{T} 
end

# manual hash and equals so that we don't simply get each struct equals missing 
function ==(a::AbstractImputedVector, b::AbstractImputedVector)
    if hash(a) != hash(b) 
        return false
    elseif (a.missingindex) != (b.missingindex)
        return false 
    elseif (a.imputedvalues) != (b.imputedvalues)
        return false 
    elseif length(a) != length(b)
        return false
    end
    for i ∈ eachindex(a)
        if a[i] != b[i]  
            return false
        end 
    end
    return true 
end

function hash(a::AbstractImputedVector, h::UInt)
    hash(
        a.original, 
        hash(a.missingindex, hash(a.imputedvalues, hash(:AbstractImputedVector, h)))
    )
end

iterate(v::AbstractImputedVector) = iterate(v, 1)

function iterate(v::AbstractImputedVector, i::Integer) 
    x = getindex(v, i)
    if i == length(v) 
        i_n = nothing 
    else 
        i_n = i + 1
    end 
    return ( x, i_n )
end 

function getindex(v::AbstractImputedVector, i::Integer) 
    if i ∈ v.missingindex
        j = findfirst(x -> x == i, v.missingindex)
        return v.imputedvalues[j, :]
    else
        return v.original[i]
    end
end

length(v::AbstractImputedVector) = length(v.original)
size(v::AbstractImputedVector) = size(v.original)
