
module SimpleMice

using AutoHashEquals, DataFrames, GLM, Random, StaticArrays, StatsBase
import Base: ^, /, *, +, -, sin, cos, tan, log, exp, getindex, isapprox, iterate, setindex!

export MiceValue
export ^, /, *, +, -, sin, cos, tan, log, exp, getindex, isapprox, iterate, setindex!
export initializemice, initializemice!, micevalues, nonemissing, updatemicevalues!

abstract type AbstractMiceValue end 

@auto_hash_equals struct MiceValue{N, T} <: AbstractMiceValue 
    x       :: MVector{N, T} 
end

function MiceValue(v::AbstractVector) 
    N = length(v)
    return MiceValue(N, v) 
end

function MiceValue(N::Integer, v::AbstractVector) 
    return MiceValue(MVector{N}(v)) 
end

getindex(a::AbstractMiceValue, i) = a.x[i]
iterate(a::AbstractMiceValue) = iterate(a.x)
iterate(a::AbstractMiceValue, i) = iterate(a.x, i)

function isapprox(a::AbstractMiceValue, b::AbstractMiceValue; kwargs...)
    return isapprox(a.x, b.x; kwargs...)
end

setindex!(a::AbstractMiceValue, v, i) = setindex!(a.x, v, i)

for f ∈ [ :^, :/, :*, :+, :- ]
    eval(quote
        function $f(a::AbstractMiceValue, b::AbstractMiceValue)
            v = @. $f(a.x, b.x)
            return MiceValue(v)
        end
        function $f(a::AbstractMiceValue, b::Number)
            v = @. $f(a.x, b)
            return MiceValue(v)
        end
        function $f(a::Number, b::AbstractMiceValue)
            v = @. $f(a, b.x)
            return MiceValue(v)
        end
    end)
end

for f ∈ [ :sin, :cos, :tan, :log, :exp ]
    eval(quote
        function $f(a::AbstractMiceValue)
            v = @. $f(a.x)
            return MiceValue(v)
        end
    end)
end

nonemissing(x::AbstractVector) = sum([ ismissing(xi) for xi ∈ x ]) == 0

function initializemice(N::Integer, x::AbstractVector)
    return initializemice(Random.default_rng(), N, x)
end

function initializemice(rng::Random.AbstractRNG, N::Integer, x::AbstractVector)
    return initializemice(rng, Float64, N, x)
end

function initializemice(T::DataType, N::Integer, x::AbstractVector)
    return initializemice(Random.default_rng(), T, N, x)
end

function initializemice(rng::Random.AbstractRNG, T::DataType, N::Integer, x::AbstractVector)
    if nonemissing(x) 
        return x 
    end
    nmvector = collect(skipmissing(x))
    @assert length(nmvector) >= 1 "Must have at least 1 non-missing value"
    Z = typeof(nmvector[1])
    return Union{Z, MiceValue{N, T}}[
        ismissing(xi) ? 
            MiceValue(N, _samplemice(rng, T, N, nmvector)) :
            xi 
        for xi ∈ x 
    ]
end

_samplemice(rng, T, N, nmvector) = [ T(sample(rng, nmvector)) for _ ∈ 1:N ]

function initializemice!(N::Integer, df::DataFrame)
    initializemice!(Random.default_rng(), N, df)
end

function initializemice!(N::Integer, df::DataFrame, i)
    initializemice!(Random.default_rng(), N, df, i)
end

function initializemice!(rng::Random.AbstractRNG, N::Integer, df::DataFrame)
    for (i, colname) ∈ enumerate(names(df))
        _initializemice!(rng, N, df, colname, i)
    end
end

function initializemice!(rng::Random.AbstractRNG, N::Integer, df::DataFrame, i::AbstractVector)
    for ix ∈ i 
        initializemice!(rng, N, df, ix)
    end
end

function initializemice!(
    rng::Random.AbstractRNG, N::Integer, df::DataFrame, colname::AbstractString
)
    if nonemissing(getproperty(df, colname)) 
        return nothing 
    end 
    i = findfirst(x -> x == colname, names(df)) 
    __initializemice!(rng, N, df, colname, i)
end

function initializemice!(rng::Random.AbstractRNG, N::Integer, df::DataFrame, colname::Symbol)
    if nonemissing(getproperty(df, colname)) 
        return nothing 
    end 
    i = findfirst(x -> x == String(colname), names(df)) 
    __initializemice!(rng, N, df, colname, i)
end

function initializemice!(rng::Random.AbstractRNG, N::Integer, df::DataFrame, i::Integer)
    colname = names(df)[i]
    _initializemice!(rng, N, df, colname, i)
end

function _initializemice!(rng, N, df, colname, i)
    if nonemissing(getproperty(df, colname)) 
        return nothing 
    end 
    __initializemice!(rng, N, df, colname, i)
end

function __initializemice!(rng, N, df, colname, i)
    newvec = initializemice(rng, N, getproperty(df, colname))
    select!(df, Not(colname))
    insertcols!(df, i, colname => newvec)
end

micevalues(x::AbstractVector, i) = [ _micevalue(xi, i) for xi ∈ x ]

function micevalues(df::DataFrame, i)
    namevec = names(df)
    return micevalues(df, namevec, i)
end

function micevalues(df::DataFrame, columnindexes::AbstractVector, i::Integer)
    M = zeros(Float64, size(df, 1), length(columnindexes))
    for (j, index) ∈ enumerate(columnindexes)
        M[:, j] .= micevalues(df, index, i)
    end
    return M
end

function micevalues(df::DataFrame, columnindex::Integer, i::Integer)
    colname = names(df)[columnindex]
    return micevalues(df, colname, i)
end

function micevalues(
    df::DataFrame, colname::S, i::Integer
) where S <: Union{<:AbstractString, Symbol}
    return micevalues(getproperty(df, colname), i)
end

_micevalue(a::AbstractMiceValue, i) = getindex(a, i)
_micevalue(a::Number, ::Integer) = a

function _linearfit(df, x, y, i)
    xmat = micevalues(df, x, i)
    yvec = micevalues(df, y, i)
    return fit(LinearModel, xmat, yvec)
end

function updatemicevalues!(
    df, y::S, x::T, N
) where {S <: Union{<:AbstractString, Symbol}, T <: Union{S, <:AbstractVector{S}}} 
    for i ∈ 1:N 
        _updatemicevalues!(df, y, x, i)
    end 
end

function updatemicevalues!(
    df, y::Integer, x::T, N
) where T <: Union{<:Integer, <:AbstractVector{<:Integer}}
    ycolname = names(df)[y]
    xcolnames = names(df)[x]
    updatemicevalues!(df, ycolname, xcolnames, N)
end

function _updatemicevalues!(df, y, x, i)
    reg = _linearfit(df, x, y, i) 
    predictions = predict(reg)
    __updatemicevalues!(df, y, predictions, i)
end

function __updatemicevalues!(df, y, predictions, i) 
    for j ∈ axes(df, 1)
        __updatemicevalues!(df, y, predictions, i, j)
    end
end

function __updatemicevalues!(df, y::S, predictions, i, j) where S <: Union{<:AbstractString, Symbol} 
    ___updatemicevalues!(df, y, getproperty(df, y)[j], predictions, i, j)
end

___updatemicevalues!(::Any, ::Any, ::Number, ::Any, ::Any, ::Any) = nothing

function ___updatemicevalues!(df, y::S, existingvalue::AbstractMiceValue, predictions, i, j) where S <: Union{<:AbstractString, Symbol} 
    getproperty(df, y)[j][i] = predictions[j]
end

end # module SimpleMice
