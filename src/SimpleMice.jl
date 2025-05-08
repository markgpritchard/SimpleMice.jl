
module SimpleMice

using AutoHashEquals: @auto_hash_equals
using Base.Threads: @threads
using DataAPI: DataAPI, describe
using DataFrames: DataFrame
using Krylov: Krylov, ktypeof
using LinearSolve: LinearSolve, KrylovJL_LSMR, LinearProblem, solve
using PrettyTables: pretty_table
using Random: AbstractRNG, default_rng
using StaticArrays: MVector, SVector
using StatsBase: StatsBase, mean, quantile, sample, var
using Tables: Tables, AbstractColumns, Schema, columnnames, getcolumn, istable, schema

export 
    ## functions from DataAPI 
    describe, 
    ## functions from StatsBase 
    mean,
    var,
    ## imputedvector.jl
    ImputedVector,
    imputedvector,
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
    elementquantile,
    isimputedvalue,
    nimputed,
    nimputedsets,
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
