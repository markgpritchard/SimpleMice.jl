
# Source for these functions: 
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2727536/

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Combinations by Rubin's rules 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
    rubinsmean(q::Vector, n::Int)

Combines the mean from each imputed dateset and produces an overall mean according 
    to Rubin's rules.

`q` is a vector of the mean from each imputed dataset and `n` is the number of imputed 
    datasets.

Note, this function does not test whether `length(q) == n`.
"""
rubinsmean(q::Vector, n::Int) = sum(q) / n

"""
    rubinsvar(q::Vector, u::Vector, n::Int)

Calculates a variance from multiple imputed datasets according to Rubin's rules. 

`q` is a vector of the mean from each imputed dataset, `u` is a vector of the variance 
    of each imputed dataset, and `n` is the number of imputed datasets.

Note, this function does not test whether `length(q) == n`.
"""
function rubinsvar(q::Vector, u::Vector, n::Int)
    ubar = withinimputationvar(u, n)
    b = betweenimputationvar(q, n)
    v = ubar + (1 + 1 / n) * b 
    return v 
end 

"""
    rubinssterror(q::Vector, u::Vector, n::Int)

Calculates a standard error from multiple imputed datasets according to Rubin's rules. 

`q` is a vector of the mean from each imputed dataset, `u` is a vector of the standard 
    error of each imputed dataset, and `n` is the number of imputed datasets.

Note, this function does not test whether `length(q) == n`.
"""
function rubinssterror(q::Vector, u::Vector, n::Int)
    ubar = withinimputationsterrorsquared(u, n)
    b = betweenimputationvar(q, n)
    v = ubar + (1 + 1 / n) * b 
    return sqrt(v) 
end 

"""
    withinimputationvar(u::Vector, n::Int) 

Calculates a mean within-imputation variance from multiple imputed datasets. 

`u` is a vector of the variance of each imputed dataset and `n` is the number of 
    imputed datasets.

Note, this function does not test whether `length(u) == n`.
"""
withinimputationvar(u::Vector, n::Int) = sum(u) / n 

"""
    betweenimputationvar(q::Vector, u::Vector, n::Int)

Calculates between-imputation variance from multiple imputed datasets. 

`q` is a vector of the mean from each imputed dataset and `n` is the number of imputed 
    datasets.

Note, this function does not test whether `length(q) == n`.
"""
function betweenimputationvar(q::Vector, n::Int)
    qbar = rubinsmean(q, n)
    qsquarediff = @. (q - qbar)^2
    b = sum(qsquarediff) / (n - 1)
    return b 
end 

"""
    withinimputationsterrorsquared(u::Vector, n::Int)

Calculates a mean within-imputation standard error squared from multiple imputed 
    datasets. 

`u` is a vector of the standard error of each imputed dataset and `n` is the number 
    of imputed datasets.

Note, this function does not test whether `length(u) == n`.
"""
withinimputationsterrorsquared(u::Vector, n::Int) = sum([ v^2 for v ∈ u ]) / n 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions used in calculating combined statistics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
    componentmeans(d::ImputedVector)

Produces a `Vector` of the mean value of each imputed vector. 
"""
componentmeans(d::ImputedVector) = componentstats(mean, d)

"""
    componentvars(d::ImputedVector)

Produces a `Vector` of the variance of each imputed vector. 
"""
componentvars(d::ImputedVector) = componentstats(var, d)

""" 
    componentstats(stat, d::ImputedVector[, <additional arguments>])

Produces a `Vector` of the statistic `stat` of each imputed vector.

`additional arguments` are passed to the function `stat`.
"""
componentstats(stat, d::ImputedVector) = [ stat(d.imputedvalues[i]) for i ∈ 1:d.numberimputed ]

function componentstats(stat, d::ImputedVector, args...)
    return [ stat(d.imputedvalues[i], args...) for i ∈ 1:d.numberimputed ]
end 

"""
    meanstats(stat, d::ImputedVector[, <additional arguments>])

Calculate the mean of the statistic `stat` calculated from each imputed dataset.

`additional arguments` are passed to the function `stat`.
"""
function meanstats(stat, d::ImputedVector)
    cs = componentstats(stat, d)
    return rubinsmean(cs, d.numberimputed)
end 

function meanstats(stat, d::ImputedVector, args...)
    cs = componentstats(stat, d, args...)
    return rubinsmean(cs, d.numberimputed)
end 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Versions of imported functions applied to imputed datasets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
    mean(d::ImputedVector)

Calculate the mean from multiple imputed datasets and combine them according to Rubin's 
    rules.
"""
mean(d::ImputedVector) = meanstats(mean, d)

"""
    var(d::ImputedVector)

Calculate the variance from multiple imputed datasets according to Rubin's rules.
"""
function var(d::ImputedVector)
    cm = componentmeans(d)
    cv = componentvars(d)
    return rubinsvar(cm, cv, d.numberimputed)
end 

"""
    std(d::ImputedVector)

Calculate the standard deviation from multiple imputed datasets as `sqrt(var(d))`. 
"""
std(d::ImputedVector) = sqrt(var(d))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Summary statistics 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function summarystats(d::ImputedVector{T}) where T <: Number
    return StatsBase.SummaryStats(
        mean(d),
        meanstats(minimum, d), # min
        meanstats(quantile, d, .25), # q25
        meanstats(median, d), # median 
        meanstats(quantile, d, .75), # q75 
        meanstats(maximum, d), # max 
        length(d.imputedvalues[1]),
        sum(ismissing.(d.imputedvalues[1] .== 1))
    )
end 

tablesummarystats(d::ImputedVector, name::String) = tablesummarystats(d, Symbol(name))

function tablesummarystats(d::ImputedVector{T}, name::Symbol) where T <: AbstractString
    elt = eltype(d)
    return ( variable = name, mean = nothing, min = minimum(d.imputedvalues[1]), 
        median = nothing, max = maximum(d.imputedvalues[1]), 
        nmissing = sum(ismissing.(d.imputedvalues[1] .== 1)), eltype = elt )
end 

function tablesummarystats(d::ImputedVector{T}, name::Symbol) where T <: Number
    stats = summarystats(d) 
    elt = eltype(d)
    return ( variable = name, mean = stats.mean, min = stats.min, 
        median = stats.median, max = stats.max, nmissing = stats.nmiss, eltype = elt )
end 

function tablesummarystats(d::ImputedVector{T}, name::Symbol) where T <: Union{<:AbstractString, Missing}
    elt = eltype(d)
    return ( variable = name, mean = nothing, min = minimum(skipmissing(d.originalvector)), 
        median = nothing, max = maximum(skipmissing(d.originalvector)), 
        nmissing = sum(ismissing.(d.originalvector .== 1)), eltype = elt )
end 

function tablesummarystats(d::ImputedVector{T}, name::Symbol) where T <: Union{<:Number, Missing}
    stats = summarystats(d.originalvector) 
    elt = eltype(d)
    return ( variable = name, mean = stats.mean, min = stats.min, 
        median = stats.median, max = stats.max, nmissing = stats.nmiss, eltype = elt )
end 
