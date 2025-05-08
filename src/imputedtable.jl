# `ImputedTable` combines vectors of non-missing and imputed values as a Table

@auto_hash_equals struct ImputedTable{Ni} <: AbstractColumns
    columnnames                 :: Vector{Symbol}
    columntypes                 :: Vector{Type}
    unchangedvectors            :: Dict{Symbol, Vector}
    imputedvectors              :: Dict{Symbol, ImputedVector}
    size1                       :: Int
end

Tables.istable(::ImputedTable) = true
Base.names(t::ImputedTable) = columnnames(t)

function Base.getproperty(t::ImputedTable, name::Symbol)
    if name ∈ [ :columnnames, :columntypes, :unchangedvectors, :imputedvectors, :size1 ]
        return getfield(t, name)
    elseif haskey(t.unchangedvectors, name)
        return t.unchangedvectors[name]
    elseif haskey(t.imputedvectors, name)
        return t.imputedvectors[name]
    else 
        return throw(ArgumentError("column name \"$name\" not found in the ImputedTable"))
    end
end

Tables.columnnames(t::ImputedTable) = getfield(t, :columnnames)
Tables.getcolumn(t::ImputedTable, i::Int) = getcolumn(t, columnnames(t)[i])
Tables.getcolumn(t::ImputedTable, name::Symbol) = getproperty(t, name)

function Tables.schema(t::ImputedTable{Ni}) where {Ni}
    return Schema(names(t), getfield(t, :columntypes))
end

Base.size(t::ImputedTable) = ( t.size1, length(t.columnnames) )
Base.size(t::ImputedTable, dim) = size(t)[dim]
Base.getindex(t::ImputedTable, rownum, colnum) = getindex(getcolumn(t, colnum), rownum)

function Base.summary(t::ImputedTable{Ni}) where Ni
    return "$(size(t, 1))×$(size(t, 2)) ImputedTable with $Ni imputations"
end

_returnimputedtableni(::ImputedTable{Ni}) where Ni = Ni

function Base.show(
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
