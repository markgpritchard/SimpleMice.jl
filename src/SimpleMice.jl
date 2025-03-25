
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
import DataAPI: describe
import Krylov: ktypeof
import StatsBase: 
    mean, 
    quantile, 
    var
import Tables: 
    AbstractColumns, 
    columnnames, 
    getcolumn, 
    istable, 
    schema

export 
    ## functions imported from DataAPI 
    describe, 
    ## functions imported from StatsBase 
    mean,
    quantile, 
    var,
    ## imputedvectorview.jl
    imputedvectorview,
    ## imputedtable.jl
    getcolumn,
    ## imputedtableview.jl
    imputedtableview,
    ## imputedtableviewmatrix.jl 
    imputedtableviewmatrix,
    vectorimputedtableviewmatrix,
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
    rubinsvar,
    ## description.jl
    nimputed,
    ## linearsolve.jl
    linearsolveimputedvector,
    linearsolveimputedvectorelements

include("consts.jl")
include("imputedvector.jl")
include("imputedvectorview.jl")
include("imputedtable.jl")
include("imputedtableview.jl")
include("imputedtableviewmatrix.jl")
include("imputedoutput.jl")
include("functions.jl")
include("impute.jl")
include("rubinsrules.jl")
include("description.jl")
include("linearsolve.jl")

end  # module SimpleMice
