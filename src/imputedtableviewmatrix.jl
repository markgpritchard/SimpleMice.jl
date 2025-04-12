
abstract type AbstractImputedTableViewMatrix{S} <: AbstractMatrix{S} end

@auto_hash_equals struct ImputedTableViewMatrix{S} <: AbstractImputedTableViewMatrix{S}
    originaltable               :: ImputedTable
    columns                     :: Vector{Int}
    index                       :: Int
    n_columns                   :: Int 
    n_rows                      :: Int

    function ImputedTableViewMatrix{S}(
        originaltable::ImputedTable{Ni}, columns, index, n_columns, n_rows
    ) where {Ni, S}
        if index > Ni || index <= 0
            throw(
                ArgumentError(
                    "Index ($index) must be positive and no greater than Ni ($Ni)"
                )
            )
        end
        if n_columns != length(columns)
            _lc = length(columns)
            throw(
                DimensionMismatch(
                    "n_columns ($n_columns) must equal the number of columns listed ($_lc)"
                )
            )
        end
        if n_rows != size(originaltable, 1) 
            _so1 = size(originaltable, 1) 
            throw(
                DimensionMismatch(
                    "n_rows ($n_rows) must equal the length of originaltable ($_so1)"
                )
            )
        end

        return new{S}(originaltable, columns, index, n_columns, n_rows)
    end
end

function ImputedTableViewMatrix(originaltable, cols, index, n_columns, n_rows)
    Tvector = _imputedtableviewtypes(originaltable, cols)
    S = __imputedtableviewtypes(Tvector[1])
    for i ∈ eachindex(Tvector)
        i == 1 && continue
        S = typeof(one(S) + one(__imputedtableviewtypes(Tvector[i])))
    end
    return ImputedTableViewMatrix{S}(originaltable, cols, index, n_columns, n_rows)
end

function ImputedTableViewMatrix(
    originaltable::ImputedTable{Ni}, cols::AbstractVector{<:Int}, index
) where {Ni}
    n_columns = length(cols)
    n_rows = size(originaltable, 1)
    return ImputedTableViewMatrix(originaltable, cols, index, n_columns, n_rows)
end

function ImputedTableViewMatrix(
    originaltable::ImputedTable, cols::AbstractVector{Symbol}, index
) 
    colindexes = _colindexes(originaltable, cols)
    return ImputedTableViewMatrix(originaltable, colindexes, index)
end

function ImputedTableViewMatrix(
    originaltable::ImputedTable, cols::AbstractVector{<:AbstractString}, index
) 
    colsymbols = Symbol.(cols)
    return ImputedTableViewMatrix(originaltable, colsymbols, index)
end

function imputedtableviewmatrix(originaltable, cols, index)
    return ImputedTableViewMatrix(originaltable, cols, index)
end

function imputedtableviewmatrix(t::ImputedTableView, cols)
    return ImputedTableViewMatrix(t.originaltable, cols, t.index)
end

@auto_hash_equals struct VectorImputedTableViewMatrix{S} <: AbstractVector{ImputedTableViewMatrix{S}}
    originaltable               :: ImputedTable
    columns                     :: Vector{Int}
    indexes                     :: Vector{Int}
    n_columns                   :: Int 
    n_rows                      :: Int

    function VectorImputedTableViewMatrix{S}(
        originaltable::ImputedTable{Ni}, columns, indexes, n_columns, n_rows
    ) where {Ni, S}
        for i ∈ indexes
            if i > Ni || i <= 0
                throw(
                    ArgumentError(
                        "Index ($i) must be positive and no greater than Ni ($Ni)"
                    )
                )
            end
        end
        if n_columns != length(columns)
            _lc = length(columns)
            throw(
                DimensionMismatch(
                    "n_columns ($n_columns) must equal the number of columns listed ($_lc)"
                )
            )
        end
        if n_rows != size(originaltable, 1) 
            _so1 = size(originaltable, 1) 
            throw(
                DimensionMismatch(
                    "n_rows ($n_rows) must equal the length of originaltable ($_so1)"
                )
            )
        end
        return new{S}(originaltable, columns, indexes, n_columns, n_rows)
    end
end

function VectorImputedTableViewMatrix(originaltable, cols, indexes, n_columns, n_rows)
    Tvector = _imputedtableviewtypes(originaltable, cols)
    S = __imputedtableviewtypes(Tvector[1])
    for i ∈ eachindex(Tvector)
        i == 1 && continue
        S = typeof(one(S) + one(__imputedtableviewtypes(Tvector[i])))
    end
    return VectorImputedTableViewMatrix{S}(originaltable, cols, indexes, n_columns, n_rows)
end

function VectorImputedTableViewMatrix(
    originaltable, cols::AbstractVector{<:Int}, indexes::AbstractVector
) 
    n_columns = length(cols)
    n_rows = size(originaltable, 1)
    return VectorImputedTableViewMatrix(originaltable, cols, indexes, n_columns, n_rows)
end

function VectorImputedTableViewMatrix(
    originaltable, cols::AbstractVector{Symbol}, indexes::AbstractVector
) 
    colindexes = _colindexes(originaltable, cols)
    return VectorImputedTableViewMatrix(originaltable, colindexes, indexes)
end

function VectorImputedTableViewMatrix(
    originaltable, cols::AbstractVector{<:AbstractString}, indexes::AbstractVector
) 
    colsymbols = Symbol.(cols)
    return VectorImputedTableViewMatrix(originaltable, colsymbols, indexes)
end

function VectorImputedTableViewMatrix(
    originaltable::ImputedTable{Ni}, cols, ::Automatic
) where {Ni}
    indexes = 1:Ni
    return VectorImputedTableViewMatrix(originaltable, cols, indexes)
end

function VectorImputedTableViewMatrix(originaltable, cols) 
    return VectorImputedTableViewMatrix(originaltable, cols, automatic) 
end

function vectorimputedtableviewmatrix(originaltable, cols, indexes=automatic)
    return VectorImputedTableViewMatrix(originaltable, cols, indexes)
end

iterate(v::ImputedTableViewMatrix) = iterate(v, 1)
iterate(::ImputedTableViewMatrix, ::Nothing) = nothing

function iterate(v::ImputedTableViewMatrix, i::Integer) 
    x = getindex(v, i)
    if i == length(v) 
        i_n = nothing 
    else 
        i_n = i + 1
    end 
    return ( x, i_n )
end 

iterate(v::VectorImputedTableViewMatrix) = iterate(v, 1)
iterate(::VectorImputedTableViewMatrix, ::Nothing) = nothing

function iterate(v::VectorImputedTableViewMatrix, i::Integer) 
    x = getindex(v, i)
    if i == length(v) 
        i_n = nothing 
    else 
        i_n = i + 1
    end 
    return ( x, i_n )
end 

_colindexes(originaltable, cols) = findall(x -> x ∈ cols, columnnames(originaltable))

__imputedtableviewtypes(::Type{Vector{T}}) where T <: Number = T
__imputedtableviewtypes(::Type{T}) where T <: Number = T

function __imputedtableviewtypes(S, T)
    return typeof(one(__imputedtableviewtypes(S))) + typeof(one(__imputedtableviewtypes(T)))
end

size(M::ImputedTableViewMatrix) = ( M.n_rows, M.n_columns )
length(M::ImputedTableViewMatrix) = M.n_rows * M.n_columns

size(v::VectorImputedTableViewMatrix) = ( length(v), )
length(v::VectorImputedTableViewMatrix) = length(v.indexes)

function getcolumn(M::ImputedTableViewMatrix, i::Int)
    ind = M.columns[i]
    return getcolumn(M.originaltable, ind)
end 

function getindex(M::ImputedTableViewMatrix, rownum, colnum)
    ind = M.columns[colnum]
    col = imputedvectorview(getproperty(M.originaltable, ind), M.index)
    return getindex(col, rownum)
end

function getindex(v::VectorImputedTableViewMatrix{S}, i) where S
    return ImputedTableViewMatrix{S}(
        v.originaltable,
        v.columns,
        i,
        v.n_columns,
        v.n_rows
    )
end
