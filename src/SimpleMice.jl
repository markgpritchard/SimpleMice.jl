
module SimpleMice

using AutoHashEquals, DataFrames, GLM, PrettyTables, Random, StaticArrays, StatsBase, Tables
import Base: ==, hash, ^, /, *, +, -, sin, cos, tan, log, exp, getindex, getproperty, 
    isapprox, iterate, length, names, setindex!, show, size, summary

export MiceArray, MiceValue, MiceVector, MiceView
export ImputedTable, ImputedTableView, ImputedVectorMStatic, ImputedVectorStatic
export ImputedVectorView, ImputedVector
export ==, hash, ^, /, *, +, -, sin, cos, tan, log, exp, getindex, getproperty, isapprox
export iterate, length, names, setindex!, show, size, summary
export impute, imputedtableview, imputedvectorview, initializemice, linearupdatemicevalues!
export micevalues, nonemissing, strictwasmissing
export updatemicevalues!, wasmissing, wasmissingindex

abstract type AbstractMiceValue{Ni} end 

@auto_hash_equals struct MiceValue{Ni, T} <: AbstractMiceValue{Ni} 
    x                   :: MVector{Ni, T} 
end

abstract type AbstractImputedVector{Ni, S, T} <: AbstractVector{S} end

struct ImputedVectorMStatic{Ni, Nm, Np, S, T} <: AbstractImputedVector{Ni, S, T}
    original                    :: Vector{<:Union{Missing, S}} 
    missingindex                :: SVector{Nm, Int64}
    imputedvalues               :: MMatrix{Nm, Ni, T, Np} 
end

struct ImputedVectorStatic{Ni, Nm, Np, S, T} <: AbstractImputedVector{Ni, S, T}
    original                    :: Vector{<:Union{Missing, S}} 
    missingindex                :: SVector{Nm, Int64}
    imputedvalues               :: SMatrix{Nm, Ni, T, Np} 
end

struct ImputedVector{Ni, Nm, Np, S, T} <: AbstractImputedVector{Ni, S, T}
    original                    :: Vector{<:Union{Missing, S}}
    missingindex                :: Vector{Int64}
    imputedvalues               :: Matrix{T} 
end

abstract type AbstractImputedVectorView{S} <: AbstractVector{S} end

@auto_hash_equals struct ImputedVectorView{S, T} <: AbstractImputedVectorView{S}
    imputedvector               :: T
    index                       :: Int
end

function ImputedVectorView(
    imputedvector::AbstractImputedVector{Ni, S, T}, index::Int
) where {Ni, S, T}
    @assert index <= Ni "Index, $index, must be no more than Ni, $Ni"
    @assert index > 0 "Index must be positive"
    combinedtype = typeof(one(S) + one(T))
    return ImputedVectorView{combinedtype, typeof(imputedvector)}(imputedvector, index)
end

function imputedvectorview(imputedvector::AbstractImputedVector, index::Int)
    return ImputedVectorView(imputedvector, index)
end 

imputedvectorview(imputedvector::AbstractVector{<:Number}, ::Int) = imputedvector

function imputedvectorview(imputedvector::AbstractVector{<:Union{Missing, <:Number}}, ::Int)
    return imputedvector
end

@auto_hash_equals struct ImputedTable{Ni} <: Tables.AbstractColumns
    columnnames                 :: Vector{Symbol}
    columntypes                 :: Vector{Type}
    unchangedvectors            :: Dict{Symbol, Vector}
    imputedmstaticvectors       :: Dict{Symbol, ImputedVectorMStatic}
    imputedvectors              :: Dict{Symbol, ImputedVector}
    size1                       :: Int
end

@auto_hash_equals struct ImputedTableView <: Tables.AbstractColumns
    originaltable               :: ImputedTable
    newcolumntypes              :: Vector{Type}
    index                       :: Int
end

function ImputedTableView(originaltable::ImputedTable{Ni}, index) where {Ni}
    @assert index <= Ni "Index, $index, must be no more than Ni, $Ni"
    @assert index > 0 "Index must be positive"
    newcolumntypes = _imputedtableviewtypes(originaltable)
    return ImputedTableView(originaltable, newcolumntypes, index)
end

imputedtableview(originaltable, index) = ImputedTableView(originaltable, index)

function _imputedtableviewtypes(originaltable::ImputedTable) 
    return [
        haskey(originaltable.unchangedvectors, name) ?
            originaltable.columntypes[i] :
            _imputedtableviewtypes(originaltable.columntypes[i])
        for (i, name) ∈ enumerate(originaltable.columnnames)
    ]
end

function _imputedtableviewtypes(::Type{<:AbstractImputedVector{Ni, S, T}}) where {Ni, S, T}
    newtype = typeof(one(S) + one(T))
    return newtype
end

_imputedtableviewtypes(::Type{Vector{T}}) where T <: Number = T

Tables.istable(::ImputedTable) = true
names(t::ImputedTable) = getfield(t, :columnnames)

function getproperty(t::ImputedTable, name::Symbol)
    if name ∈ [ 
        :columnnames, 
        :columntypes, 
        :unchangedvectors, 
        :imputedmstaticvectors, 
        :imputedvectors, 
        :size1 
    ]
        return getfield(t, name)
    elseif haskey(t.unchangedvectors, name)
        return t.unchangedvectors[name]
    elseif haskey(t.imputedmstaticvectors, name)
        return t.imputedmstaticvectors[name]
    elseif haskey(t.imputedvectors, name)
        return t.imputedvectors[name]
    else 
        return throw(ArgumentError("column name \"$name\" not found in the ImputedTable"))
    end
end

Tables.columnnames(t::ImputedTable) = getfield(t, :columnnames)
Tables.getcolumn(t::ImputedTable, i::Int) = Tables.getcolumn(t, Tables.columnnames(t)[i])
Tables.getcolumn(t::ImputedTable, name::Symbol) = getproperty(t, name)

function Tables.schema(t::ImputedTable{Ni}) where {Ni}
    return Tables.Schema(names(t), getfield(t, :columntypes))
end

size(t::ImputedTable) = ( t.size1, length(t.columnnames) )
size(t::ImputedTable, dim) = size(t)[dim]
getindex(t::ImputedTable, rownum, colnum) = getindex(Tables.getcolumn(t, colnum), rownum)

function summary(t::ImputedTable{Ni}) where Ni
    return "$(size(t, 1))×$(size(t, 2)) ImputedTable with $Ni imputations"
end

function summary(t::ImputedTableView) 
    Ni = _returnimputedtableni(t.originaltable)
    return "$(size(t, 1))×$(size(t, 2)) ImputedTableView, imputation $(t.index) of $Ni"
end

_returnimputedtableni(t::ImputedTable{Ni}) where Ni = Ni

function show(
    io::IO, t::ImputedTable; 
    crop_subheader=true, 
    hlines=[ 1 ], 
    minimum_columns_width=7, 
    show_row_number=true, 
    title=summary(t), 
    vlines=[ 1 ], 
    kwargs...
)
    return pretty_table(
        io, t; 
        crop_subheader, hlines, minimum_columns_width, show_row_number, title, vlines, 
        kwargs...
    )
end

function show(
    io::IO, t::ImputedTableView; 
    crop_subheader=true, 
    hlines=[ 1 ], 
    minimum_columns_width=7, 
    show_row_number=true, 
    title=summary(t), 
    vlines=[ 1 ], 
    kwargs...
)
    return pretty_table(
        io, t; 
        crop_subheader, hlines, minimum_columns_width, show_row_number, title, vlines, 
        kwargs...
    )
end

Tables.istable(::ImputedTableView) = true
names(t::ImputedTableView) = names(t.originaltable)

function getproperty(t::ImputedTableView, name::Symbol)
    if name ∈ [ :originaltable, :newcolumntypes, :index ]
        return getfield(t, name)
    else
        return imputedvectorview(getproperty(t.originaltable, name), t.index)
    end
end

Tables.columnnames(t::ImputedTableView) = Tables.columnnames(t.originaltable)
Tables.getcolumn(t::ImputedTableView, i::Int) = Tables.getcolumn(t, Tables.columnnames(t)[i])
Tables.getcolumn(t::ImputedTableView, name::Symbol) = getproperty(t, name)
Tables.schema(t::ImputedTableView) = Tables.Schema(names(t), getfield(t, :newcolumntypes))

size(t::ImputedTableView) = size(t.originaltable)
size(t::ImputedTableView, dim) = size(t)[dim]
getindex(t::ImputedTableView, rownum, colnum) = getindex(Tables.getcolumn(t, colnum), rownum)
show(io::IO, t::ImputedTableView) = pretty_table(t)

# manual hash and equals so that we don't simply get each struct equals missing 
function ==(a::AbstractImputedVector, b::AbstractImputedVector)
    if hash(a) != hash(b) 
        return false
    elseif (a.missingindex) != (b.missingindex)
        return false 
    elseif (a.imputedvalues) != (b.imputedvalues)
        return false 
    elseif length(a) != length(b)
        return false
    end
    for i ∈ eachindex(a)
        if a[i] != b[i]  
            return false
        end 
    end
    return true 
end

function hash(a::AbstractImputedVector, h::UInt)
    hash(
        a.original, 
        hash(a.missingindex, hash(a.imputedvalues, hash(:AbstractImputedVector, h)))
    )
end

iterate(v::AbstractImputedVector) = iterate(v, 1)

function iterate(v::AbstractImputedVector, i::Integer) 
    x = getindex(v, i)
    if i == length(v) 
        i_n = nothing 
    else 
        i_n = i + 1
    end 
    return ( x, i_n )
end 

function getindex(v::AbstractImputedVector, i::Integer) 
    if i ∈ v.missingindex
        j = findfirst(x -> x == i, v.missingindex)
        return v.imputedvalues[j, :]
    else
        return v.original[i]
    end
end

length(v::AbstractImputedVector) = length(v.original)
size(v::AbstractImputedVector) = size(v.original)

iterate(v::AbstractImputedVectorView) = iterate(v, 1)

function iterate(v::AbstractImputedVectorView, i::Integer) 
    x = getindex(v, i)
    if i == length(v) 
        i_n = nothing 
    else 
        i_n = i + 1
    end 
    return ( x, i_n )
end 

iterate(::AbstractImputedVectorView, ::Nothing) = nothing

function getindex(v::ImputedVectorView{S, T}, i::Integer) where {S, T}
    if i ∈ v.imputedvector.missingindex
        j = findfirst(x -> x == i, v.imputedvector.missingindex)
        return S(v.imputedvector.imputedvalues[j, v.index])
    else
        return S(v.imputedvector.original[i])
    end
end

length(v::AbstractImputedVectorView) = length(v.imputedvector)
size(v::AbstractImputedVectorView) = size(v.imputedvector)

function MiceValue(v::AbstractVector) 
    M = length(v)
    return MiceValue(M, v) 
end

function MiceValue(M::Integer, v::AbstractVector) 
    return MiceValue(MVector{M}(v)) 
end

function isapprox(a::AbstractMiceValue, b::AbstractMiceValue; kwargs...)
    return isapprox(a.x, b.x; kwargs...)
end

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

function initializemice(Ni::Integer, args...)
    return initializemice(Random.default_rng(), Ni, args...)
end

function initializemice(rng::Random.AbstractRNG, Ni::Integer, args...)
    return initializemice(rng, Float64, Ni, args...)
end

function initializemice(T::DataType, Ni::Integer, args...)
    return initializemice(Random.default_rng(), T, Ni, args...)
end

function initializemice(rng::Random.AbstractRNG, T::DataType, Ni::Integer, v::AbstractVector)
    nmvector = collect(skipmissing(v))
    @assert length(nmvector) >= 1 "Must have at least 1 non-missing value"
    S = typeof(nmvector[1])
    Nm = sum(ismissing.(v))
    Np = Ni * Nm 
    if Np == 0 
        return v
    elseif Np < 100 
        return _initializemicestatic(rng, Ni, Nm, Np, S, T, v, nmvector)
    else 
        return _initializemicenotstatic(rng, Ni, Nm, Np, S, T, v, nmvector)
    end
end

function initializemice(rng::Random.AbstractRNG, T::DataType, Ni::Integer, df::DataFrame)
    columnnames = Symbol.(names(df))
    columntypes = Vector{Type}(undef, length(columnnames))                 
    unchangedvectors::Dict{Symbol, Vector} = Dict()
    imputedmstaticvectors::Dict{Symbol, ImputedVectorMStatic} = Dict()
    imputedvectors::Dict{Symbol, ImputedVector} = Dict()
    size1 = size(df, 1)
    for (j, name) ∈ enumerate(columnnames) 
        vnew = initializemice(rng, T, Ni, getproperty(df, name))
        _pushinitializemice!(
            unchangedvectors, imputedmstaticvectors, imputedvectors, name, vnew
        )
        columntypes[j] = typeof(vnew)
    end
    return ImputedTable{Ni}(
        columnnames, 
        columntypes, 
        unchangedvectors, 
        imputedmstaticvectors, 
        imputedvectors, 
        size1
    )
end

function initializemice(
    rng::Random.AbstractRNG, T::DataType, Ni::Integer, df::DataFrame, column::AbstractString
)
    return initializemice(rng, T, Ni, df, [ column ])
end

function initializemice(
    rng::Random.AbstractRNG, T::DataType, Ni::Integer, df::DataFrame, column::Symbol
)
    return initializemice(rng, T, Ni, df, [ column ])
end

function initializemice(
    rng::Random.AbstractRNG, T::DataType, Ni::Integer, df::DataFrame, column::Integer
)
    return initializemice(rng, T, Ni, df, [ column ])
end

function initializemice(
    rng::Random.AbstractRNG, 
    T::DataType, 
    Ni::Integer, 
    df::DataFrame, 
    columns::Vector{<:AbstractString}
)
    return initializemice(rng, T, Ni, df, Symbol.(columns))
end

function initializemice(
    rng::Random.AbstractRNG, 
    T::DataType, 
    Ni::Integer, 
    df::DataFrame, 
    columns::AbstractVector{<:Integer}
)
    passcolumnnames = Symbol.(names(df))[columns]
    return initializemice(rng, T, Ni, df, passcolumnnames)
end

function initializemice(
    rng::Random.AbstractRNG, T::DataType, Ni::Integer, df::DataFrame, columns::Vector{Symbol}
)
    columnnames = Symbol.(names(df))
    columntypes = Vector{Type}(undef, length(columnnames))                 
    unchangedvectors::Dict{Symbol, Vector} = Dict()
    imputedmstaticvectors::Dict{Symbol, ImputedVectorMStatic} = Dict()
    imputedvectors::Dict{Symbol, ImputedVector} = Dict()
    size1 = size(df, 1)
    for (j, name) ∈ enumerate(columnnames) 
        if name ∈ columns
            vnew = initializemice(rng, T, Ni, getproperty(df, name))
            _pushinitializemice!(
                unchangedvectors, imputedmstaticvectors, imputedvectors, name, vnew
            )
            columntypes[j] = typeof(vnew)
        else
            push!(unchangedvectors, name => getproperty(df, name))
            columntypes[j] = typeof(getproperty(df, name))
        end
    end
    return ImputedTable{Ni}(
        columnnames, 
        columntypes, 
        unchangedvectors, 
        imputedmstaticvectors, 
        imputedvectors, 
        size1
    )
end

function _pushinitializemice!(
    unchangedvectors, imputedmstaticvectors, imputedvectors, name, vnew::Vector{<:Number}
)
    push!(unchangedvectors, name => vnew)
end

function _pushinitializemice!(
    unchangedvectors, imputedmstaticvectors, imputedvectors, name, vnew::ImputedVectorMStatic
)
    push!(imputedmstaticvectors, name => vnew)
end

function _pushinitializemice!(
    unchangedvectors, imputedmstaticvectors, imputedvectors, name, vnew::ImputedVector
)
    push!(imputedvectors, name => vnew)
end

function _initializemicestatic(rng, Ni, Nm, Np, S, T, v, nmvector)
    return ImputedVectorMStatic{Ni, Nm, Np, S, T}(
        v,
        SVector{Nm, Int}(findall(ismissing, v)),
        MMatrix{Nm, Ni, T, Np}([ T(sample(rng, nmvector)) for _ ∈ 1:Nm, _ ∈ 1:Ni ])
    )
end

function _initializemicenotstatic(rng, Ni, Nm, Np, S, T, v, nmvector)
    return ImputedVector{Ni, Nm, Np, S, T}(
        v,
        findall(ismissing, v),
        [ T(sample(rng, nmvector)) for _ ∈ 1:Nm, _ ∈ 1:Ni ]
    )
end

_samplemice(rng, T, N, nmvector) = [ T(sample(rng, nmvector)) for _ ∈ 1:N ]

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

_micevalue(a::Number, ::Integer) = a

function linearupdatemicevalues!(table::ImputedTableView, y::Symbol, x::Vector{Symbol})
    lhs = Term(y)
    rhsterm = Term.(x)
    rhs = ( ConstantTerm(1), rhsterm... )
    formula = FormulaTerm(lhs, rhs)
    regr = fit(LinearModel, formula, table)
    predictions = predict(regr)
    #println(predictions)
    for (i, j) ∈ enumerate(getproperty(table.originaltable, y).missingindex)
        getproperty(table.originaltable, y).imputedvalues[i, table.index] = predictions[j]
    end
end

function linearupdatemicevalues!(
    table::ImputedTableView, y::AbstractString, x::Vector{<:AbstractString}
)
    return linearupdatemicevalues!(table, Symbol(y), Symbol.(x))
end

function linearupdatemicevalues!(table::ImputedTableView, y::Integer, x::Vector{<:Integer})
    yname = table.originaltable.columnnames[y]
    xnames = [ table.originaltable.columnnames[xi] for xi ∈ x ]
    return linearupdatemicevalues!(table, yname, xnames)
end

function linearupdatemicevalues!(table::ImputedTableView, y::S, x::S) where S
    return linearupdatemicevalues!(table, y, [ x ])
end

function linearupdatemicevalues!(
    table::ImputedTable{Ni}, iteratevars, includevars, iterations
) where Ni
    Threads.@threads for i ∈ 1:Ni 
        tableview = imputedtableview(table, i)
        for _ ∈ 1:iterations
            for v ∈ iteratevars 
                x = [ iteratevars[findall(z -> z != v, iteratevars)]; includevars ]
                linearupdatemicevalues!(tableview, v, x)
            end
        end
    end
end

function impute(Ni::Integer, df, iteratevars, includevars, iterations)
    return impute(Random.default_rng(), Ni, df, iteratevars, includevars, iterations)
end

function impute(
    rng::Random.AbstractRNG, Ni::Integer, df, iteratevars, includevars, iterations
)
    return impute(rng, Float64, Ni, df, iteratevars, includevars, iterations)
end

function impute(T::DataType, Ni::Integer, df, iteratevars, includevars, iterations)
    return impute(Random.default_rng(), T, Ni, df, iteratevars, includevars, iterations)
end

function impute(
    rng::Random.AbstractRNG, 
    T::DataType, 
    Ni::Integer, 
    df, 
    iteratevars::Vector, 
    includevars::Vector, 
    iterations
)
    table = initializemice(rng, T, Ni, df, [ iteratevars; includevars ])
    linearupdatemicevalues!(table, iteratevars, includevars, iterations)
    return table 
end

function impute(
    rng::Random.AbstractRNG, 
    T::DataType, 
    Ni::Integer, 
    df, 
    iteratevars, 
    includevars::Vector, 
    iterations
)
    return impute(rng, T, Ni, df, [ iteratevars ], includevars, iterations) 
end

function impute(
    rng::Random.AbstractRNG, 
    T::DataType, 
    Ni::Integer, 
    df, 
    iteratevars, 
    includevars, 
    iterations
)
    return impute(rng, T, Ni, df, iteratevars, [ includevars ], iterations) 
end

function impute(
    rng::Random.AbstractRNG, 
    T::DataType, 
    Ni::Integer, 
    df, 
    iteratevars::AbstractUnitRange, 
    includevars::Vector, 
    iterations
)
    return impute(rng, T, Ni, df, collect(iteratevars), includevars, iterations) 
end

function impute(
    rng::Random.AbstractRNG, 
    T::DataType, 
    Ni::Integer, 
    df, 
    iteratevars, 
    includevars::AbstractUnitRange, 
    iterations
)
    return impute(rng, T, Ni, df, iteratevars, collect(includevars), iterations) 
end

end  # module SimpleMice
