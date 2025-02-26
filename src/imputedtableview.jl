# `ImputedTableView` accesses values from an `ImputedTable`, giving all original non-missing
# values and one of the imputed values for each missing value

@auto_hash_equals struct ImputedTableView <: AbstractColumns
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

istable(::ImputedTableView) = true
names(t::ImputedTableView) = columnnames(t)

function getproperty(t::ImputedTableView, name::Symbol)
    if name ∈ [ :originaltable, :newcolumntypes, :index ]
        return getfield(t, name)
    else
        return imputedvectorview(getproperty(t.originaltable, name), t.index)
    end
end

columnnames(t::ImputedTableView) = columnnames(t.originaltable)
getcolumn(t::ImputedTableView, i::Int) = getcolumn(t, columnnames(t)[i])
getcolumn(t::ImputedTableView, name::Symbol) = getproperty(t, name)
schema(t::ImputedTableView) = Schema(names(t), getfield(t, :newcolumntypes))

size(t::ImputedTableView) = size(t.originaltable)
size(t::ImputedTableView, dim) = size(t)[dim]
getindex(t::ImputedTableView, rownum, colnum) = getindex(getcolumn(t, colnum), rownum)

function summary(t::ImputedTableView) 
    Ni = _returnimputedtableni(t.originaltable)
    return "$(size(t, 1))×$(size(t, 2)) ImputedTableView, imputation $(t.index) of $Ni"
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
