
module SimpleMice

using AutoHashEquals, DataFrames, GLM, Random, StaticArrays, StatsBase
import Base: ^, /, *, +, -, sin, cos, tan, log, exp, getindex, isapprox, iterate, length, setindex!, size

export MiceArray, MiceValue, MiceVector, MiceView
export ^, /, *, +, -, sin, cos, tan, log, exp, getindex, isapprox, iterate, length, setindex!, size
export initializemice, initializemice!, micevalues, nonemissing, strictwasmissing
export updatemicevalues!, wasmissing, wasmissingindex

abstract type AbstractMiceValue{M} end 

@auto_hash_equals struct MiceValue{M, T} <: AbstractMiceValue{M} 
    x                   :: MVector{M, T} 
end

@auto_hash_equals struct MiceArray{M, S, T, N} <: AbstractArray{Union{T, MiceValue{M, S}}, N} 
    v                   :: Array{Union{T, MiceValue{M, S}}, N} 
end

const MiceVector{M, S, T} = MiceArray{M, S, T, 1}

@auto_hash_equals struct MiceView{M, Z, S, T, N} <: AbstractArray{Z, N}
    parent              :: MiceArray{M, S, T, N} 
    iteration           :: Int
    wasmissingindex     :: Vector{Int}
end

function MiceValue(v::AbstractVector) 
    M = length(v)
    return MiceValue(M, v) 
end

function MiceValue(M::Integer, v::AbstractVector) 
    return MiceValue(MVector{M}(v)) 
end

function MiceView(parent::MiceArray{M, S, T, N}, iteration) where {M, S, T, N}
    iteration > M && throw(
        DimensionMismatch(
            "cannot form a view for iteration greater than the number of imputed values"
        )
    )
    wmi = wasmissingindex(parent)
    return _makemiceview(parent, iteration, wmi)
end

# keep this code, but at the moment it is quicker to use the version below 
#=
function MiceView(parent::MiceVector{M, S, T}, iteration, j) where {M, S, T}
    iteration > M && throw(
        DimensionMismatch(
            "cannot form a view for iteration greater than the number of imputed values"
        )
    )
    wmi = wasmissingindex(parent[j])
    return _makemiceview(MiceVector{M, S, T}(parent[j]), iteration, wmi)
end

function MiceView(parent::MiceVector{M, S, T}, iteration, j::Integer) where {M, S, T}
    iteration > M && throw(
        DimensionMismatch(
            "cannot form a view for iteration greater than the number of imputed values"
        )
    )
    wmi = wasmissingindex([parent[j]])
    return _makemiceview(MiceVector{M, S, T}([parent[j]]), iteration, wmi)
end
=#

function MiceView(parent::MiceVector{M, S, T}, iteration, j) where {M, S, T}
    a = MiceView(parent, iteration)
    return a[j] 
end



function _makemiceview(parent::MiceArray{M, S, S, N}, iteration, wmi) where {M, S, N}
    return MiceView{M, S, S, S, N}(parent, iteration, wmi)
end

function _makemiceview(parent::MiceArray{M, S, T, N}, iteration, wmi) where {M, S, T, N}
    Z = typeof(one(S) + one(T))
    return MiceView{M, Z, S, T, N}(parent, iteration, wmi)
end

getindex(a::AbstractMiceValue, i) = a.x[i]
iterate(a::AbstractMiceValue) = iterate(a.x)
iterate(a::AbstractMiceValue, i) = iterate(a.x, i)
getindex(a::MiceArray, i::Integer) = getindex(a.v, i)
getindex(a::MiceArray, args...) = getindex(a.v, args...)
iterate(a::MiceArray) = iterate(a.v)
iterate(a::MiceArray, i) = iterate(a.v, i)
length(a::MiceArray) = length(a.v)
size(a::MiceArray) = size(a.v)

function getindex(a::MiceView{M, Z, S, T, N}, i::Integer) where {M, Z, S, T, N}
    if i ∈ a.wasmissingindex 
        return _micevalue(a, i, :a)
    else 
        return _micevalue(a, i)
    end 
end

function getindex(a::MiceView{M, Z, S, T, N}, i) where {M, Z, S, T, N}
    return [ getindex(a, ix) for ix ∈ i ]
end

iterate(a::MiceView) = iterate(a.parent)
iterate(a::MiceView, i) = iterate(a.parent, i)
length(a::MiceView) = length(a.parent)
size(a::MiceView) = size(a.parent)

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

_warntime::Float64 = 1.74e9

wasmissing(::AbstractMiceValue; kwargs...) = true

function wasmissing(::Missing; warn=true)
    if warn && time() - _warntime > 10 
        @warn """
        `wasmissing` called on missing value. Returns `true`. 
        This warning is shown at most once every 10 seconds.
        """
        global _warntime = time() 
    end 
    return true 
end 

wasmissing(::Any; kwargs...) = false

strictwasmissing(::AbstractMiceValue) = true
strictwasmissing(::Any) = false
wasmissingindex(v::AbstractVector) = findall(strictwasmissing, v)

function initializemice(M::Integer, x::AbstractVector)
    return initializemice(Random.default_rng(), M, x)
end

function initializemice(rng::Random.AbstractRNG, M::Integer, x::AbstractVector)
    return initializemice(rng, Float64, M, x)
end

function initializemice(T::DataType, M::Integer, x::AbstractVector)
    return initializemice(Random.default_rng(), T, M, x)
end

function initializemice(rng::Random.AbstractRNG, S::DataType, M::Integer, x::AbstractVector)
    if nonemissing(x) 
        return x 
    end
    nmvector = collect(skipmissing(x))
    @assert length(nmvector) >= 1 "Must have at least 1 non-missing value"
    T = typeof(nmvector[1])
    return MiceVector{M, S, T}(
        [
            ismissing(xi) ? 
                MiceValue(M, _samplemice(rng, S, M, nmvector)) :
                xi 
            for xi ∈ x 
        ]
    )
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

function initializemice!(
    rng::Random.AbstractRNG, N::Integer, df::DataFrame, i::AbstractVector
)
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
    println(typeof(newvec))
    select!(df, Not(colname))
    insertcols!(df, i, colname => newvec)
end

micevalues(x::AbstractVector, i) = [ _micevalue(xi, i) for xi ∈ x ]

#=
function getindex(a::MiceView{T, P, N}, i) where {T, P, N}
    if i ∈ a.wasmissingindex 
        x = _micevalue(getindex(a.parent, i), a.iteration) 
    else 
        x = getindex(a.parent, i) 
    end 
    return T(x)
end
=#


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
_micevalue(x::MiceView{M, Z, S, T, N}, i) where {M, Z, S, T, N} = Z(getindex(x.parent, i))

function _micevalue(x::MiceView{M, Z, S, T, N}, i, ::Symbol) where {M, Z, S, T, N}
    return Z(_micevalue(getindex(x.parent, i), x.iteration))
end

function _linearfit(df, x, y, i)
    xmat = micevalues(df, x, i)
    yvec = micevalues(df, y, i)
    return _linearfit(df, xmat, yvec)
end

function _linearfit(df, xvec::Vector{S}, yvec::Vector) where S
    m = length(xvec)
    xmat = Matrix{S}(undef, m, 1)
    xmat[:, 1] .= xvec
    return _linearfit(df, xmat, yvec)
end

_linearfit(df, xmat::Matrix, yvec::Vector) = fit(LinearModel, xmat, yvec)

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

function __updatemicevalues!(
    df, y::S, predictions, i, j
) where S <: Union{<:AbstractString, Symbol} 
    ___updatemicevalues!(df, y, getproperty(df, y)[j], predictions, i, j)
end

___updatemicevalues!(::Any, ::Any, ::Number, ::Any, ::Any, ::Any) = nothing

function ___updatemicevalues!(
    df, y::S, existingvalue::AbstractMiceValue, predictions, i, j
) where S <: Union{<:AbstractString, Symbol} 
    getproperty(df, y)[j][i] = predictions[j]
end

end  # module SimpleMice
