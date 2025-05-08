# Functions to combine imputed values. Soure for Rubin's rules:
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2727536/

abstract type ImputedResult{Ni, T} end 

@auto_hash_equals struct ImputedSum{Ni, T} <: ImputedResult{Ni, T}
    elementsums                 :: SVector{Ni, T}
end

@auto_hash_equals struct ImputedMean{Ni, T} <: ImputedResult{Ni, T}
    elementmeans                :: SVector{Ni, T}
    summarymean                 :: T
end

@auto_hash_equals struct ImputedVariance{Ni, T} <: ImputedResult{Ni, T}
    mean                        :: ImputedMean{Ni, T} 
    elementvars                 :: SVector{Ni, T}
    withinimputationvariance    :: T 
    betweenimputationvariance   :: T
    totalvariance               :: T
end

Base.iterate(v::ImputedSum) = iterate(v, 1)
Base.iterate(v::ImputedSum, i) = iterate(v.elementsums, i) 
Base.getindex(v::ImputedSum, i) = getindex(v.elementsums, i)
Base.length(v::ImputedSum) = length(v.elementsums)
Base.size(v::ImputedSum) = size(v.elementsums)
Base.:(==)(a::ImputedSum, b::AbstractVector) = a.elementsums == b
Base.:(==)(a::AbstractVector, b::ImputedSum) = a == b.elementsums

function Base.show(io::IO, ::MIME"text/plain", a::ImputedVariance{Ni, T}) where {Ni, T}
    print(
        io,
        """
        Variance of imputed dataset with $Ni imputed versions
        $(a.summaryvar)
        $(a.elementvars)
        """
    )
end

function Base.show(io::IO, ::MIME"text/plain", a::ImputedMean{Ni, T}) where {Ni, T}
    print(
        io,
        """
        Mean of imputed dataset with $Ni imputed versions
        $(a.summarymean)
        $(a.elementmeans)
        """
    )
end

function Base.sum(v::AbstractImputedVector{Ni, S, T}) where {Ni, S, T}
    return ImputedSum{Ni, _typeofsum(S, T)}([ sum(imputedvectorview(v, i)) for i ∈ 1:Ni ])
end

function _elementmean(v::AbstractImputedVector{Ni, S, T}) where {Ni, S, T}
    return SVector{Ni, _typeofsumdivided(S, T)}(
        [ mean(imputedvectorview(v, i)) for i ∈ 1:Ni ]
    )
end

function imputedmean(v::AbstractImputedVector{Ni, S, T}) where {Ni, S, T}
    elementmeans = _elementmean(v)
    return _imputedmean(Ni, elementmeans)
end

function _imputedmean(Ni, elementmeans)
    summarymean = sum(elementmeans) / Ni
    return ImputedMean(elementmeans, summarymean)
end

displayelementmeans(m::ImputedMean) = m.elementmeans
displayelementmeans(m::ImputedVariance) = displayelementmeans(m.mean)
meanvalue(m::ImputedMean) = m.summarymean
meanvalue(m::ImputedVariance) = meanvalue(m.mean)

function StatsBase.mean(v::AbstractImputedVector)
    m = imputedmean(v)
    return meanvalue(m)
end

function _elementvar(v::AbstractImputedVector{Ni, S, T}; kwargs...) where {Ni, S, T}
    return SVector{Ni, _typeofsumdivided(S, T)}(
        [ var(imputedvectorview(v, i); kwargs...) for i ∈ 1:Ni ]
    )
end

function betweenimputationvar(v; kwargs...)
    m = imputedmean(v)
    return betweenimputationvar(m; kwargs...)
end

function betweenimputationvar(m::ImputedMean{Ni, T}; corrected=true) where {Ni, T}
    vec = [ (m.elementmeans[i] - m.summarymean)^2 for i ∈ 1:Ni ] 
    numer = sum(vec)
    denom = Ni - Int(corrected)
    return numer / denom
end

function imputedvar(v::AbstractImputedVector; mean=nothing, kwargs...)
    return _imputedvar(v, mean; kwargs...)
end

function _imputedvar(v::AbstractImputedVector, ::Nothing; kwargs...)
    m = imputedmean(v) 
    return _imputedvar(v, m; kwargs...)
end

function _imputedvar(v::AbstractImputedVector, mean::Number; kwargs...)
    m = imputedmean(v) 
    @assert meanvalue(m) == mean 
    return _imputedvar(v, m; kwargs...)
end

function _imputedvar(
    v::AbstractImputedVector{Ni, S, T}, mean::ImputedMean{Ni, U}; 
    kwargs...
) where {Ni, S, T, U}
    elementwisevariances = _elementvar(v; kwargs...)
    withinimputationvariance = sum(elementwisevariances) / Ni
    betweenimputationvariance = betweenimputationvar(mean; kwargs...)
    totalvariance = withinimputationvariance + betweenimputationvariance * (1 + 1 / Ni)
    return ImputedVariance(
        mean, 
        elementwisevariances, 
        withinimputationvariance, 
        betweenimputationvariance, 
        totalvariance
    )
end

displayelementvars(m::ImputedVariance) = m.elementvars
varvalue(m::ImputedVariance) = m.totalvariance

function StatsBase.var(v::AbstractImputedVector; kwargs...)
    v = imputedvar(v; kwargs...)
    return varvalue(v)
end
