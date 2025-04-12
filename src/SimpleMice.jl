
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
    var,
    ## imputedvector.jl
    ImputedVector,
    ## imputedvectorview.jl
    ImputedVectorView,
    imputedvectorview,
    ## imputedtable.jl
    ImputedTable,
    getcolumn,
    ## imputedtableview.jl
    ImputedTableView,
    imputedtableview,
    ## imputedtableviewmatrix.jl 
    ImputedTableViewMatrix,
    VectorImputedTableViewMatrix,
    imputedtableviewmatrix,
    vectorimputedtableviewmatrix,
    ## functions.jl
    nonemissing,
    ## impute.jl
    impute,
    initializemice,
    initializemicevector,
    linearupdatemicevalues!,
    ## rubinsrules.jl
    betweenimputationvar,
    displayelementmeans,
    displayelementvars,
    elementvar,
    imputedmean,
    imputedvar,
    meanvalue,
    rubinsvar,
    varvalue,
    ## description.jl
    isimputedvalue,
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
include("functions.jl")
include("impute.jl")
include("rubinsrules.jl")
include("description.jl")
include("linearsolve.jl")

end  # module SimpleMice
