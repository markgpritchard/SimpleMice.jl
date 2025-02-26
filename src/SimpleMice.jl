
module SimpleMice

using GLM, StaticArrays, StatsBase
using Base.Threads: @threads
using AutoHashEquals: @auto_hash_equals
using DataFrames: DataFrame
using PrettyTables: pretty_table
using Random: AbstractRNG, default_rng
using Tables: Schema

import Base: 
    ==, 
    ^, 
    /, 
    *, 
    +, 
    -, 
    cos, 
    exp, 
    getindex, 
    getproperty, 
    hash, 
    isapprox, 
    iterate, 
    length, 
    log,
    names, 
    setindex!, 
    show, 
    sin, 
    size, 
    summary,
    tan
import Tables: AbstractColumns, columnnames, getcolumn, istable, schema

export 
    ## imputedvectorview.jl
    imputedvectorview,
    ## imputedtableview.jl
    imputedtableview,
    ## functions.jl
    nonemissing,
    ## impute.jl
    impute,
    initializemice,
    linearupdatemicevalues!

include("micevalue.jl")
include("imputedvector.jl")
include("imputedvectorview.jl")
include("imputedtable.jl")
include("imputedtableview.jl")
include("functions.jl")
include("impute.jl")

end  # module SimpleMice
