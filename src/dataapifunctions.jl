
# No documentation added for this function as already fully described in ?describe(a)
function describe(io::IO, a::AbstractImputedData) 
    show(io, summarystats(a))
    println(io, "Type:           $(string(eltype(a)))")
end

function describe(io::IO, a::ImputedVector{T})  where T <: AbstractString
    println(io, "Summary Stats:")
    println(io, "Length:         $(length(a.imputedvalues[1]))")
    println(io, "Type:           $(string(eltype(a)))")
    println(io, "Number Unique:  $(length(unique(a.imputedvalues[1])))")
    return
end

"""
    describe(idf::ImputedDataFrame; vars = names(idf.originaldf))

Return descriptive statistics for each variable in `vars` in an `ImputedDataFrame` 
    as a new `DataFrame`.
"""
function describe(idf::ImputedDataFrame; vars = names(idf.originaldf))
    output = DataFrame([
        :variable => Symbol[], 
        :mean => Union{Float64, Nothing}[],
        :min => Any[],
        :median => Union{Float64, Nothing}[],
        :max => Any[],
        :nmissing => Int[],
        :eltype => Type[]
    ])
    for var ∈ vars 
        push!(output, tablesummarystats(getvalues(idf, var), var))
    end 
    return output
end 
