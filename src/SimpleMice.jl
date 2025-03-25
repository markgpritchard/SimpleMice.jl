
module SimpleMice

using LinearSolve, StaticArrays, StatsBase
using Base.Threads: @threads
using AutoHashEquals: @auto_hash_equals
using DataFrames: DataFrame
using PrettyTables: pretty_table
using Random: AbstractRNG, default_rng
using Tables: Schema

import Base: 
    ==, 
    getindex, 
    getproperty, 
    hash, 
    isapprox, 
    iterate, 
    length, 
    names, 
    setindex!, 
    show, 
    size, 
    sum,
    summary
import Krylov: ktypeof
import StatsBase: mean, var
import Tables: AbstractColumns, columnnames, getcolumn, istable, schema

export 
    ## functions imported from StatsBase 
    mean,
    var,
    ## imputedvectorview.jl
    imputedvectorview,
    ## imputedtableview.jl
    imputedtableview,
    ## imputedtableviewmatrix.jl 
    imputedtableviewmatrix,
    ## imputedoutput.jl
    summaryimputedoutput,
    ## functions.jl
    nonemissing,
    ## impute.jl
    impute,
    initializemice,
    linearupdatemicevalues!,
    ## rubinsrules.jl
    elementmean,
    elementvar,
    rubinsvar

include("imputedvector.jl")
include("imputedvectorview.jl")
include("imputedtable.jl")
include("imputedtableview.jl")
include("imputedtableviewmatrix.jl")
include("imputedoutput.jl")
include("functions.jl")
include("impute.jl")
include("rubinsrules.jl")

end  # module SimpleMice
