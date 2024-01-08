
module SimpleMice

#using DataFrames, Distributions, GLM, Random, StaticArrays, StatsBase
using DataFrames, GLM, Random, StaticArrays, StatsBase
using HypothesisTests: confint, OneSampleTTest, pvalue
import Base: /, *, +, -
#import DataAPI: describe
#import StatsBase: mean, std, summarystats, var
import StatsBase: var
import StatsModels: TableRegressionModel

include("types.jl")
include("mice.jl")
#include("classifyvariables.jl")
include("basefunctions.jl")
include("statsfunctions.jl")
include("testdataset.jl")

export 
    # types.jl
    AbstractImputedData, ImputedData, ImputedMissingData, ImputedNonMissingData, ImputedRegressionResult,
    # mice.jl 
    mice, mice!,
    # basefunctions.jl
    /, *, +, -,
    # statsfunctions.jl
    imputedlm, rubinsmean, rubinsvar, var

end # module SimpleMice
