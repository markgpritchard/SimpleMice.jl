
module SimpleMice

using AutoHashEquals: @auto_hash_equals
using Base.Threads: @threads
using DataAPI: DataAPI, describe
using DataFrames: DataFrame
using Krylov: Krylov, ktypeof
using LinearSolve: LinearSolve, KrylovJL_LSMR, LinearProblem, solve
using PrettyTables: pretty_table
using Random: AbstractRNG, default_rng
using Reexport: @reexport
using StaticArrays: MVector, SVector
using StatsBase: StatsBase, mean, quantile, sample, var
using Tables: Tables, AbstractColumns, Schema, columnnames, getcolumn, istable, schema

@reexport using DataAPI: describe
@reexport using StatsBase: mean, var
@reexport using Tables: getcolumn

## imputedvector.jl
export ImputedVector, imputedvector
## imputedvectorview.jl
export ImputedVectorView, imputedvectorview
## imputedtable.jl
export ImputedTable
## imputedtableview.jl
export ImputedTableView, imputedtableview
## imputedtableviewmatrix.jl 
export ImputedTableViewMatrix, VectorImputedTableViewMatrix
export imputedtableviewmatrix, vectorimputedtableviewmatrix
## functions.jl
export nonemissing
## impute.jl
export impute, initializemice, initializemicevector, linearupdatemicevalues!
## rubinsrules.jl
export betweenimputationvar
export displayelementmeans
export displayelementvars
export elementvar
export imputedmean
export imputedvar
export meanvalue
export rubinsvar
export varvalue
## description.jl
export elementquantile, isimputedvalue, nimputed, nimputedsets
## linearsolve.jl
export linearsolveimputedvector, linearsolveimputedvectorelements

struct Automatic end  # not exported
const automatic = Automatic()  # not exported 

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
