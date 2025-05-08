# `ImputedTableView` accesses values from an `ImputedTable`, giving all original non-missing
# values and one of the imputed values for each missing value

"""
    ImputedTableView

A table that allows functions to access one imputed dataset from an `originaltable` produced 
    by multiple imputation.

New `ImputedTableView` are expected to be created with the function `imputedtableview`.

# Fields
- `originaltable::ImputedTable`: original multiple imputation table
- `newcolumntypes::Vector{Type}`: type for each column in the table
- `index::Int`: index, `1 <= index <= Ni`, for the imputed values included in this view
"""
@auto_hash_equals struct ImputedTableView <: AbstractColumns
    originaltable::ImputedTable
    newcolumntypes::Vector{Type}
    index::Int

    function ImputedTableView(
        originaltable::ImputedTable{Ni}, newcolumntypes, index
    ) where Ni
        0 < index <= Ni || throw(_imputedtableviewerror(index, Ni))
        return new(originaltable, newcolumntypes, index)
    end
end

function ImputedTableView(originaltable::ImputedTable{Ni}, index) where {Ni}
    newcolumntypes = _imputedtableviewtypes(originaltable)
    return ImputedTableView(originaltable, newcolumntypes, index)
end

"""
    imputedtableview(originaltable, index)

Create a `ImputedTableView` from an `ImputedTable`
"""
imputedtableview(originaltable, index) = ImputedTableView(originaltable, index)

function _imputedtableviewtypes(originaltable::ImputedTable) 
    return _imputedtableviewtypes(originaltable, originaltable.columnnames) 
end

function _imputedtableviewtypes(originaltable::ImputedTable, columns::AbstractVector{<:Int}) 
    colnames = originaltable.columnnames[columns]
    return _imputedtableviewtypes(originaltable, colnames) 
end

function _imputedtableviewtypes(originaltable::ImputedTable, colnames::Vector{Symbol}) 
    return [
        haskey(originaltable.unchangedvectors, name) ?
            originaltable.columntypes[i] :
            _imputedtableviewtypes(originaltable.columntypes[i])
        for (i, name) ∈ enumerate(colnames)
    ]
end

function _imputedtableviewtypes(::Type{<:AbstractImputedVector{Ni, S, T}}) where {Ni, S, T}
    newtype = typeof(one(S) + one(T))
    return newtype
end

_imputedtableviewtypes(::Type{Vector{T}}) where T <: Number = T

Tables.istable(::ImputedTableView) = true
Base.names(t::ImputedTableView) = columnnames(t)

function Base.getproperty(t::ImputedTableView, name::Symbol)
    if name ∈ [ :originaltable, :newcolumntypes, :index ]
        return getfield(t, name)
    else
        return imputedvectorview(getproperty(t.originaltable, name), t.index)
    end
end

Tables.columnnames(t::ImputedTableView) = columnnames(t.originaltable)
Tables.getcolumn(t::ImputedTableView, i::Int) = getcolumn(t, columnnames(t)[i])
Tables.getcolumn(t::ImputedTableView, name::Symbol) = getproperty(t, name)
Tables.schema(t::ImputedTableView) = Schema(names(t), getfield(t, :newcolumntypes))

Base.size(t::ImputedTableView) = size(t.originaltable)
Base.size(t::ImputedTableView, dim) = size(t)[dim]
Base.getindex(t::ImputedTableView, rownum, colnum) = getindex(getcolumn(t, colnum), rownum)

function Base.summary(t::ImputedTableView) 
    Ni = _returnimputedtableni(t.originaltable)
    return "$(size(t, 1))×$(size(t, 2)) ImputedTableView, imputation $(t.index) of $Ni"
end

function Base.show(
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

function _imputedtableviewerror(index, Ni)
    ArgumentError("Index ($index) must be positive and no greater than Ni ($Ni)")
end
