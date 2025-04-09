# functions to describe imputed data

nimputed(v::AbstractImputedVector) = v.Nm
nimputed(v::AbstractImputedVectorView) = nimputed(v.imputedvector)
nimputed(::AbstractVector) = 0

isimputedvalue(v::AbstractImputedVector, i) = i ∈ v.missingindex
isimputedvalue(v::AbstractImputedVectorView, i) = isimputedvalue(v.imputedvector, i)

function elementquantile(v::AbstractImputedVector{Ni, S, T}, p; kwargs...) where {Ni, S, T}
    Z = typeof(one(S) + one(T))
    return SVector{Ni, Z}([ quantile(imputedvectorview(v, i), p; kwargs...) for i ∈ 1:Ni ])
end

function quantile(v::AbstractImputedVector{Ni, S, T}, p; kwargs...) where {Ni, S, T}
    return sum(elementquantile(v, p; kwargs...)) / Ni
end

function quantile(v::AbstractImputedVector, ps::Vector; kwargs...) 
    return [ quantile(v, p; kwargs...) for p ∈ ps ]
end

function describe(t::ImputedTable; cols=columnnames(t))
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
