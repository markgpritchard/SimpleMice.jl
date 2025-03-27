# Functions to combine imputed values. Soure for Rubin's rules:
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2727536/

abstract type ImputedResult{Ni, T} end 

@auto_hash_equals struct ImputedSum{Ni, T} <: ImputedResult{Ni, T}
    elementsums     :: SVector{Ni, T}
end

@auto_hash_equals struct ImputedMean{Ni, T} <: ImputedResult{Ni, T}
    elementmeans    :: SVector{Ni, T}
    summarymean     :: T
end

@auto_hash_equals struct ImputedVariance{Ni, T} <: ImputedResult{Ni, T}
    elementmeans    :: SVector{Ni, T}
    summarymean     :: T
    elementvars     :: SVector{Ni, T}
    summaryvar      :: T
end

iterate(v::ImputedSum) = iterate(v, 1)
iterate(v::ImputedSum, i) = iterate(v.elementsums, i) 
getindex(v::ImputedSum, i) = getindex(v.elementsums, i)
length(v::ImputedSum) = length(v.elementsums)
size(v::ImputedSum) = size(v.elementsums)
==(a::ImputedSum, b::AbstractVector) = a.elementsums == b
==(a::AbstractVector, b::ImputedSum) = a == b.elementsums

_typeofsum(S, T) = typeof(one(S) + one(T))
_typeofsumdivided(S, T) = typeof(one(S) + one(T) / 1)

function show(io::IO, ::MIME"text/plain", a::ImputedVariance{Ni, T}) where {Ni, T}
    print(
        io,
        """
        Variance of imputed dataset with $Ni imputed versions
        $(a.summaryvar)
        $(a.elementvars)
        """
    )
end

function show(io::IO, ::MIME"text/plain", a::ImputedMean{Ni, T}) where {Ni, T}
    print(
        io,
        """
        Mean of imputed dataset with $Ni imputed versions
        $(a.summarymean)
        $(a.elementmeans)
        """
    )
end

function sum(v::AbstractImputedVector{Ni, S, T}) where {Ni, S, T}
    return ImputedSum{Ni, _typeofsum(S, T)}([ sum(imputedvectorview(v, i)) for i ∈ 1:Ni ])
end

function elementmean(v::AbstractImputedVector{Ni, S, T}) where {Ni, S, T}
    return SVector{Ni, _typeofsumdivided(S, T)}(
        [ mean(imputedvectorview(v, i)) for i ∈ 1:Ni ]
    )
end

elementmean(m::ImputedMean) = m.elementmeans
elementmean(v::ImputedVariance) = v.elementmeans

function imputedmean(v::AbstractImputedVector{Ni, S, T}) where {Ni, S, T}
    elementmeans = elementmean(v)
    return _imputedmean(Ni, elementmeans)
end

function _imputedmean(Ni, elementmeans)
    m = sum(elementmeans) / Ni
    return ImputedMean{Ni, typeof(m)}(elementmeans, m)
end

function mean(v::AbstractImputedVector)
    m = imputedmean(v)
    return mean(m)
end

mean(m::ImputedMean) = m.summarymean
mean(v::ImputedVariance) = v.summarymean

function elementvar(v::AbstractImputedVector{Ni, S, T}; corrected=true) where {Ni, S, T}
    return SVector{Ni, _typeofsumdivided(S, T)}(
        [ var(imputedvectorview(v, i); corrected) for i ∈ 1:Ni ]
    )
end

elementvar(v::ImputedVariance) = v.elementvars

"""
    imputedvar(v::AbstractImputedVector; corrected::Bool=true, mean=nothing)

Compute the variance of a vector containing imputed values according to Rubin's rules. 
    Returns the result in an `ImputedVariance` struct.
"""
function imputedvar(v::AbstractImputedVector; corrected::Bool=true, mean=nothing)
    return _imputedvar(v, mean; corrected)
end

function _imputedvar(v::AbstractImputedVector, ::Nothing; kwargs...)
    imputedmeanv = imputedmean(v) 
    return _imputedvar(v, imputedmeanv; kwargs...)
end

function _imputedvar(v::AbstractImputedVector, mean::Number; kwargs...)
    imputedmeanv = imputedmean(v) 
    # passing the mean is intended to aid efficiency so this assertion appears contrary to
    # that, but imputedmeanv needs to be calculated so once we have two possible values for
    # the mean I want to check they're the same
    @assert imputedmeanv.summarymean == mean 
    return _imputedvar(v, imputedmeanv; kwargs...)
end

function _imputedvar(
    v::AbstractImputedVector{Ni, S, T}, mean::ImputedMean; 
    corrected=true
) where {Ni, S, T}
    elementvars = elementvar(v; corrected)
    summaryvar = rubinsvar(Ni, mean, elementvars; corrected)
    return ImputedVariance(mean.elementmeans, mean.summarymean, elementvars, summaryvar)
end

function rubinsvar(Ni, elementmeans::AbstractVector, elementvars; kwargs...)
    imputedmean = _imputedmean(Ni, elementmeans)
    return rubinsvar(Ni, imputedmean, elementvars; kwargs...)
end

function rubinsvar(Ni, imputedmean::ImputedMean, elementvars; corrected=true)
    return _rubinsvar(Ni, imputedmean, elementvars, corrected)
end

function _rubinsvar(Ni, imputedmean, elementvars, corrected::Bool)
    ubar = mean(elementvars)
    qsquarediff = @. (imputedmean.elementmeans - imputedmean.summarymean)^2
    b = sum(qsquarediff) / (Ni - Int(corrected)) 
    return ubar + (1 + 1 / Ni) * b
end

"""
    var(v::AbstractImputedVector; corrected::Bool=true, mean=nothing)

Compute the variance of a vector containing imputed values according to Rubin's rules. 
"""
function var(v::AbstractImputedVector; corrected::Bool=true, mean=nothing)
    v = imputedvar(v; corrected, mean)
    return var(v)
end

var(m::ImputedVariance) = m.summaryvar
