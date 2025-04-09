# Each `AbstractImputedVector` type gives a vector that combines original non-missing values
# (of type `S`) and a series of `Ni` imputed values (of type `T`)

abstract type AbstractImputedVector{Ni, S, T} <: AbstractVector{S} end  # not exported

struct ImputedVector{Ni, S, T} <: AbstractImputedVector{Ni, S, T}  # not exported
    original                    :: Vector{<:Union{Missing, S}}
    missingindex                :: Vector{Int}
    imputedvalues               :: Vector{MVector{Ni, T}}
    Nm                          :: Int 
    Np                          :: Int

    function ImputedVector{Ni, S, T}(
        original, missingindex, imputedvalues, Nm, Np
        ) where {Ni, S, T}
        if Ni * Nm != Np 
            throw(
                DimensionMismatch("Ni ($Ni) must be the product of Ni ($Ni) and Nm ($Nm)")
            )
        else 
            return new{Ni, S, T}(original, missingindex, imputedvalues, Nm, Np)
        end
    end
end

# manual hash and equals so that we don't simply get each struct equals missing 
function ==(a::AbstractImputedVector, b::AbstractImputedVector)
    if hash(a) != hash(b) 
        return false
    elseif a.Nm != b.Nm 
        return false 
    elseif a.Np != b.Np 
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

function hash(a::ImputedVector{Ni, S, T}, h::UInt) where {Ni, S, T}
    hash(
        Ni,
        hash(
            a.original, 
            hash(
                a.missingindex, 
                hash(
                    a.imputedvalues, 
                    hash(
                        a.Nm,
                        hash(
                            a.Np,
                            hash(
                                :AbstractImputedVector, 
                                h
                            )
                        )
                    )
                )
            )
        )
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
    if isimputedvalue(v, i)
        j = findfirst(x -> x == i, v.missingindex)
        return v.imputedvalues[j]
    else
        return v.original[i]
    end
end

length(v::AbstractImputedVector) = length(v.original)
size(v::AbstractImputedVector) = size(v.original)
