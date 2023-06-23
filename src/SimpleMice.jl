
module SimpleMice

using DataFrames, Distributions, GLM, PrettyTables, Random, StatsBase
using HypothesisTests: confint, OneSampleTTest, pvalue
import StatsBase: mean, std, var
import GLM: fit, glm, lm

include("types.jl")
include("constants.jl")
include("mice.jl")
include("classifyvariables.jl")
include("basefunctions.jl")
include("statsfunctions.jl")
include("glmfunctions.jl")

export mice
export getvalues,
    betweenimputationvar, rubinsmean, rubinsvar, withinimputationvar,
    componentmeans, componentvars,
    mean, std, var,
    fit, glm, lm

end
