# contains general functions used to create or manipulate datasets

nonemissing(x::AbstractVector) = sum([ ismissing(xi) for xi ∈ x ]) == 0

_inputsymbol(a::Symbol) = a 
_inputsymbol(::Any, a::Symbol) = a 
_inputsymbol(a::AbstractString) = Symbol(a)
_inputsymbol(::Any, a::AbstractString) = Symbol(a)
_inputsymbol(names, a::Integer) = _inputsymbol(names[a])
_inputsymbol(df::DataFrame, a::Integer) = _inputsymbol(names(df)[a])
_inputsymbol(t::ImputedTable, a::Integer) = _inputsymbol(names(t)[a])
_inputsymbol(t::ImputedTableView, a::Integer) = _inputsymbol(names(t)[a])

_inputsymbolvector(v::Vector{Symbol}) = v
_inputsymbolvector(::Any, v::Vector{Symbol}) = v
_inputsymbolvector(v::Vector{<:AbstractString}) = Symbol.(v)
_inputsymbolvector(::Any, v::Vector{<:AbstractString}) = Symbol.(v)
_inputsymbolvector(names, v::Vector{<:Integer}) = _inputsymbolvector(names[v])
_inputsymbolvector(df::DataFrame, v::Vector{<:Integer}) = _inputsymbolvector(names(df)[v])
_inputsymbolvector(t::ImputedTable, v::Vector{<:Integer}) = _inputsymbolvector(names(t)[v])
_inputsymbolvector(t::ImputedTableView, v::Vector{<:Integer}) = _inputsymbolvector(names(t)[v])
_inputsymbolvector(names, v::AbstractUnitRange) = _inputsymbolvector(names, collect(v))
_inputsymbolvector(a::Symbol) = _inputsymbolvector([ a ])
_inputsymbolvector(a::AbstractString) = _inputsymbolvector([ a ])
_inputsymbolvector(::Any, a::Symbol) = _inputsymbolvector([ a ])
_inputsymbolvector(::Any, a::AbstractString) = _inputsymbolvector([ a ])
_inputsymbolvector(names, a::Integer) = _inputsymbolvector(names, [ a ])
