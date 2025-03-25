
abstract type AbstractImputedTableViewMatrix{S} <: AbstractMatrix{S} end

@auto_hash_equals struct ImputedTableViewMatrix{S} <: AbstractImputedTableViewMatrix{S}
    originaltable               :: ImputedTable
    columns                     :: Vector{Int}
    index                       :: Int
    n_columns                   :: Int 
    n_rows                      :: Int
end

function ImputedTableViewMatrix(originaltable::ImputedTable{Ni}, cols::AbstractVector{<:Int}, index) where {Ni}
    @assert index <= Ni "Index, $index, must be no more than Ni, $Ni"
    @assert index > 0 "Index, $index, must be positive"
    n_columns = length(cols)
    n_rows = size(originaltable, 1)
    Tvector = _imputedtableviewtypes(originaltable, cols)
    #println("Tvector = $Tvector")
    S = __imputedtableviewtypes(Tvector[1])
    for i ∈ eachindex(Tvector)
        i == 1 && continue
        S = typeof(one(S) + one(__imputedtableviewtypes(Tvector[i])))
    end
    return ImputedTableViewMatrix{S}(originaltable, cols, index, n_columns, n_rows)
end

function ImputedTableViewMatrix(originaltable::ImputedTable, cols::AbstractVector{Symbol}, index) 
    colindexes = _colindexes(originaltable, cols)
    return ImputedTableViewMatrix(originaltable, colindexes, index)
end

function ImputedTableViewMatrix(originaltable::ImputedTable, cols::AbstractVector{<:AbstractString}, index) 
    colsymbols = Symbol.(cols)
    return ImputedTableViewMatrix(originaltable, colsymbols, index)
end

function imputedtableviewmatrix(originaltable, cols, index)
    return ImputedTableViewMatrix(originaltable, cols, index)
end

function imputedtableviewmatrix(t::ImputedTableView, cols)
    return ImputedTableViewMatrix(t.originaltable, cols, t.index)
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

_colindexes(originaltable, cols) = findall(x -> x ∈ cols, columnnames(originaltable))

__imputedtableviewtypes(::Type{Vector{T}}) where T <: Number = T
__imputedtableviewtypes(::Type{T}) where T <: Number = T
__imputedtableviewtypes(S, T) = typeof(one(__imputedtableviewtypes(S))) + typeof(one(__imputedtableviewtypes(T)))

size(M::ImputedTableViewMatrix) = ( M.n_rows, M.n_columns )
length(M::ImputedTableViewMatrix) = M.n_rows * M.n_columns

function getcolumn(M::ImputedTableViewMatrix, i::Int)
    ind = M.columns[i]
    return getcolumn(M.originaltable, ind)
end 

function getindex(M::ImputedTableViewMatrix, rownum, colnum)
    ind = M.columns[colnum]
    col = imputedvectorview(getproperty(M.originaltable, ind), M.index)
    return getindex(col, rownum)
end
