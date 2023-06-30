
"""
    oneach(func, <function arguments>; <keyword arguments>)

Applies `func` to each imputed `DataFrame` in an `ImputedDataFrame`, or to each result 
    in an `ImputedResult` structure. 

One of the `function arguments` must be an `ImputedDataFrame` or an `ImputedResult` 
    structure. The function is applied to each imputed result and a vector of these 
    outputs is provided in an `ImputedResult` structure. `Keyword arguments` are all 
    passed to `func`.
"""
function oneach(func, args...; kwargs...)
    return _oneach(func, args; kwargs...)
end 

function _oneach(func, args::Tuple; kwargs...)
    idfposition = findfirst(x -> isa(x, ImputedDataFrame), args)
    return _oneach(func, idfposition, args; kwargs...)
end 

function _oneach(func, idfposition::Nothing, args; kwargs...)
    irposition = findfirst(x -> isa(x, ImputedResult), args)
    @assert !isnothing(irposition) "arguments to one each must include an ImputedDataFrame or OnEachResult"
    return _oneach(func, irposition, args; kwargs...)
end 

function _oneach(func, idfposition::Int, args; kwargs...)
    idf = args[idfposition]
    return _oneach(func, idf, idfposition, args; kwargs...)
end 

function _oneach(func, idf::ImputedDataFrame, idfposition, args; kwargs...)
    newargs = [ j == idfposition ? idf.imputeddfs[1] : arg for (j, arg) ∈ enumerate(args) ]
    results1 = func(newargs...; kwargs...)
    return _oneach(func, results1, idf, idfposition, args; kwargs...)
end 

function _oneach(func, results1::T, idf::ImputedDataFrame, idfposition, args; kwargs...) where T
    results = Vector{T}(undef, idf.numberimputed)
    results[1] = results1
    for i ∈ eachindex(results)
        i == 1 && continue
        newargs = [ j == idfposition ? idf.imputeddfs[i] : arg for (j, arg) ∈ enumerate(args) ]
        results[i] = func(newargs...; kwargs...)
    end 
    return ImputedResult{T}(results, idf.numberimputed)
end 

function _oneach(func, ir::ImputedResult, irposition, args; kwargs...)
    newargs = [ j == irposition ? ir.results[1] : arg for (j, arg) ∈ enumerate(args) ]
    results1 = func(newargs...; kwargs...)
    return _oneach(func, results1, ir, irposition, args; kwargs...)
end 

function _oneach(func, results1::T, ir::ImputedResult, irposition, args; kwargs...) where T
    results = Vector{T}(undef, ir.n)
    results[1] = results1
    for i ∈ eachindex(results)
        i == 1 && continue
        newargs = [ j == irposition ? ir.results[i] : arg for (j, arg) ∈ enumerate(args) ]
        results[i] = func(newargs...; kwargs...)
    end 
    return ImputedResult{T}(results, ir.n)
end 

"""
    function desentinelize!(df::DataFrame)
        
Remove `SentinelArrays` from `DataFrame`.

Based on code shared at 
    https://discourse.julialang.org/t/disable-sentinelarrays-for-csv-read/54843/3
"""
function desentinelize!(df::DataFrame)
    for (i, col) ∈ enumerate(eachcol(df)) df[!, i] = collect(col) end
end
