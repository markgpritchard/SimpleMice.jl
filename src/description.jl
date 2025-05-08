# functions to describe imputed data

nimputed(v::AbstractImputedVector) = v.Nm
nimputed(v::AbstractImputedVectorView) = nimputed(v.imputedvector)
nimputed(::AbstractVector) = 0

nimputedsets(::AbstractImputedVector{Ni, S, T}) where {Ni, S, T} = Ni
nimputedsets(v::AbstractImputedVectorView) = nimputedsets(v.imputedvector)
nimputedsets(::AbstractVector) = 0

isimputedvalue(v::AbstractImputedVector, i) = i ∈ v.missingindex
isimputedvalue(v::AbstractImputedVectorView, i) = isimputedvalue(v.imputedvector, i)

function elementquantile(v::AbstractImputedVector{Ni, S, T}, p; kwargs...) where {Ni, S, T}
    Z = typeof((one(S) + one(T)) / 1)
    return SVector{Ni, Z}([ quantile(imputedvectorview(v, i), p; kwargs...) for i ∈ 1:Ni ])
end

"""
    quantile(v::AbstractImputedVector, p; <keyword arguments>)

Returns the arithmetic mean of the quantile(s) at probability `p` calculated for each 
    imputed dataset.

Keyword arguments are used for calculating the quantile(s) in each imputed dataset.

See also `elementquantile`.
"""
function StatsBase.quantile(v::AbstractImputedVector, p; kwargs...)
    return sum(elementquantile(v, p; kwargs...)) / nimputedsets(v)
end

function StatsBase.quantile(v::AbstractImputedVector, ps::Vector; kwargs...) 
    return [ quantile(v, p; kwargs...) for p ∈ ps ]
end

function DataAPI.describe(t::ImputedTable; cols=columnnames(t))
    outputdf = DataFrame(
        :variable => Symbol[ ],
        :mean => Float64[ ],
        :min => Float64[ ],
        :median => Float64[ ], 
        :max => Float64[ ], 
        :nmissing => Int[ ], 
        :nimputed => Int[ ], 
        :eltype => Type[ ]
    )
    for c ∈ cols 
        ss = summarystats(getproperty(t, c))
        newlinedf = DataFrame(
            :variable => c,
            :mean => ss.mean,
            :min => ss.min,
            :median => ss.median, 
            :max => ss.max, 
            :nmissing => ss.nmiss, 
            :nimputed => nimputed(getproperty(t, c)), 
            :eltype => eltype(getproperty(t, c))
        )
        append!(outputdf, newlinedf)
    end
    return outputdf
end
