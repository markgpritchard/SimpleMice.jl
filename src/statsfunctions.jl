
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Values from imputed value type 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

getvalue(a::ImputedNonMissingData, i) = a.v 
getvalue(a::ImputedMissingData, i) = a.v[i] 
#getvalue(a::ImputedMissingContData, i) = a.v[i] 
#getvalue(a::ImputedMissingOrderedData, i) = a.v[i] 
#getvalue(a::ImputedMissingUnOrderedData, i) = a.categories[a.v[i]]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Combinations by Rubin's rules 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Source for these functions: 
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2727536/

"""
    rubinsmean(vec::Vector{<:ImputedData{N, T}}) where {N, T}

Combines the mean from each imputed dateset and produces an overall mean according 
    to Rubin's rules.
"""
function rubinsmean(vec::Vector{<:ImputedData{N, T}}) where {N, T}
    m = mean(vec)
    t = sum([ getvalue(m, i) for i ∈ 1:N ])
    return t / N 
end 

function var(vec::Vector{<:ImputedData{N, T}}) where {N, T}
    result = MVector{N}( [ var([ getvalue(a, i) for a ∈ vec ]) for i ∈ 1:N ])
    return ImputedMissingData(result)
end

"""
    rubinsvar(q::Vector, u::Vector, n::Int)

Calculates a variance from multiple imputed datasets according to Rubin's rules. 

`q` is a vector of the mean from each imputed dataset, `u` is a vector of the variance 
    of each imputed dataset, and `n` is the number of imputed datasets.

Note, this function does not test whether `length(q) == n`.
"""
function rubinsvar(vec::Vector{<:ImputedData{N, T}}) where {N, T}
    m = mean(vec)
    rm = rubinsmean(vec)
    vr = var(vec)
    vw = sum([ getvalue(vr, i) for i ∈ 1:N ]) / N
    vb = sum([ (getvalue(m, i) - rm)^2 for i ∈ 1:N ]) / (N - 1)
    vtotal = vw + vb * (1 + 1 / N) 
    return vtotal 
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Regressions  
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#=
function imputedfit(model, formula, data, distr; kwargs...)

end
=#

function imputedlm(formula, data; kwargs...)
    data_n = size(data, 1)
    # find N 
    N = _findN(formula, data)
    rawresults = rawimputedlm(formula, data, N; kwargs...)
    cn = coefnames(rawresults[1])
    allcoefs, allsterrors = imputedlmresultmatrix(rawresults, length(cn), N)
    coefs = [ mean(allcoefs[i]) for i ∈ eachindex(cn) ]
    vtotals = imputedlmrubinsvar(coefs, allcoefs, allsterrors, length(cn), N)
    t, p, ci = imputedlmttests(coefs, vtotals, data_n) 
    lmdf = imputedlmdf(cn, coefs, vtotals, t, p, ci)
    println(lmdf)
    return ImputedRegressionResult(cn, allcoefs, allsterrors, coefs, vtotals, t, p, ci, lmdf)
end

function _findN(formula, data)
    N = _findN(getproperty(data, Symbol(formula.lhs)))
    return _findN(N, formula, data, 1)
end

_findN(v::Vector{<:Number}) = nothing

_findN(v::Vector{<:ImputedData{N, T}}) where {N, T} = N

_findN(N::Int, formula, data, i) = N

function _findN(N::Nothing, formula, data, i)
    x = formula.rhs[i]
    if isa(x, ConstantTerm) 
        return _findN(nothing, formula, data, i + 1)
    else 
        N = _findN(getproperty(data, Symbol(x)))
        return _findN(N, formula, data, i + 1)
    end
end

function rawimputedlm(formula, data, N; kwargs...)
    # Create a temp DataFrame 
    tdf = createimputeddf(formula, data)
    # Create a vector to store the results  
    rawresults = Vector{TableRegressionModel}(undef, N)
    # Run the first regression 
    rawresults[1] = lm(formula, tdf; kwargs...)
    # Repeat for all other imputed datasets
    for i ∈ 2:N 
        mutateimputeddf!(tdf, formula, data, i)
        rawresults[i] = lm(formula, tdf; kwargs...)
    end 
    return rawresults
end

function formulacolumns(formula)
    lhs_symbol = Symbol(formula.lhs) 
    rhs_index = findall(x -> isa(x, Term), formula.rhs)
    rhs_symbol = Symbol.(formula.rhs[rhs_index])
    cols = [ lhs_symbol; [ x for x ∈ rhs_symbol ]]
    return cols
end

function createimputeddf(data, formula::FormulaTerm, i = 1; kwargs...)
    cols = formulacolumns(formula)
    return createimputeddf(data, cols, i; kwargs...) 
end

function createimputeddf(data, columns::Vector{<:Symbol}, i = 1; kwargs...)
    tdf = DataFrame()
    for x ∈ columns insertcols!(tdf, x => tempdfcol(data, x, i; kwargs...)) end
    return tdf
end

function tempdfcol(data, v, i; orderedcatvars = Symbol[ ], unorderedcatvars = Symbol[ ]) 
    newv = tempdfcol(getproperty(data, v), i)
    if v ∈ orderedcatvars || v ∈ unorderedcatvars 
        oldv = getproperty(data, v) 
        levelsknown = false 
        k = 1
        while !levelsknown 
            if isa(oldv[k], ImputedMissingData) levelsknown = true 
            else                                k += 1
            end 
        end
        levels = oldv[k].levels
        newv = categorical(newv; levels, ordered = v ∈ orderedcatvars)
    end
    return newv
end

tempdfcol(v::Vector{<:Real}, i) = v 

tempdfcol(v::Vector{<:ImputedData{N, T}}, i) where {N, T} = T[ getvalue(x, i) for x ∈ v ]

tempdfcol(v::CategoricalVector{<:ImputedData{N, T}}, i) where {N, T} = T[ getvalue(x, i) for x ∈ v ]

function mutateimputeddf!(tdf, data, formula::FormulaTerm, i)
    cols = formulacolumns(formula)
    mutateimputeddf!(tdf, data, columns, i)
end

function mutateimputeddf!(tdf, data, columns::Vector{<:Symbol}, i)
    for x ∈ columns tempdfcol!(tdf, x, data, i) end
end

tempdfcol!(tdf, v, data::DataFrame, i) = tempdfcol!(tdf, v, getproperty(data, v), i)

tempdfcol!(tdf, v, vals::Vector{<:Real}, i) = nothing

function tempdfcol!(tdf, v, vals::Vector{<:ImputedData}, i) 
    for j ∈ axes(tdf, 1) getproperty(tdf, v)[j] = getvalue(vals[j], i) end
end

function imputedlmresultmatrix(rawresults, ℓ, N)
    allcoefs = MArray{Tuple{ℓ, N}, Float64}(undef)
    allsterrors = MArray{Tuple{ℓ, N}, Float64}(undef)
    for i ∈ 1:N 
        df = DataFrame(coeftable(rawresults[i]))
        allcoefs[:, i] = getproperty(df, Symbol("Coef."))
        allsterrors[:, i] = getproperty(df, Symbol("Std. Error"))
    end 
    return ( allcoefs, allsterrors )
end

function imputedlmrubinsvar(coefs, allcoefs, allsterrors, ℓ, N)
    vw = [ mean(allsterrors[i]) for i ∈ 1:ℓ ]
    vb = zeros(ℓ)
    for j ∈ 1:ℓ
        vb[j] = sum([ (allcoefs[j, i] - coefs[j])^2 for i ∈ 1:N ]) / (N - 1)
    end
    return vw + vb * (1 + 1 / N) 
end

function imputedlmttests(coefs, vtotals, data_n) 
    ttests = [ OneSampleTTest(coefs[i], vtotals[i] * sqrt(data_n), data_n) for i ∈ eachindex(coefs) ]
    t = [ x.t for x ∈ ttests ]
    p = [ pvalue(x) for x ∈ ttests ]
    ci = [ confint(x) for x ∈ ttests ]
    return ( t, p, ci )
end

imputedlmdf(cn, coefs, vtotals, t, p, ci) = DataFrame(
    Symbol("") => cn,
    Symbol("Coef.") => coefs,
    Symbol("Std. Error") => vtotals,
    :t => t,
    Symbol("Pr(>|t|)") => p, 
    Symbol("Lower 95%") => [ c[1] for c ∈ ci ],
    Symbol("Upper 95%") => [ c[2] for c ∈ ci ]
)
