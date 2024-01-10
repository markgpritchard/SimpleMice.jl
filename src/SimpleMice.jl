
module SimpleMice

using DataFrames, GLM, Random, StaticArrays, StatsBase
using HypothesisTests: confint, OneSampleTTest, pvalue
import Base: ==, /, *, +, -
import StatsBase: var
import StatsModels: TableRegressionModel

include("types.jl")
include("basefunctions.jl")
include("mice.jl")
#include("classifyvariables.jl")
include("statsfunctions.jl")

export 
    # types.jl
    AbstractImputedData, ImputedData, ImputedMissingBoolData, ImputedMissingData, 
    ImputedNonMissingData, ImputedRegressionResult,
    # basefunctions.jl
    ==, /, *, +, -,
    # mice.jl 
    mice, mice!,
    # statsfunctions.jl
    imputedlm, rubinsmean, rubinsvar, var

end # module SimpleMice
