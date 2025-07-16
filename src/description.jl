# functions to describe imputed data

"""
    nimputed(v)

The number of imputed values in a vector
"""
nimputed(v::AbstractImputedVectorView) = nimputed(v.imputedvector)
nimputed(::AbstractVector) = 0

"""
    nimputedsets(v)

The number of imputed values calculated for each value that was originally missing.
"""
nimputedsets(v::AbstractImputedVectorView) = nimputedsets(v.imputedvector)
nimputedsets(::AbstractVector) = 0

isimputedvalue(v::AbstractImputedVectorView, i) = isimputedvalue(v.imputedvector, i)



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
