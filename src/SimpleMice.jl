
module SimpleMice

using DataFrames, Distributions, GLM, Random, StatsBase
using HypothesisTests: confint, OneSampleTTest, pvalue
import Base: eltype
import DataAPI: describe
import StatsBase: mean, std, summarystats, var
import GLM: fit, glm, lm

include("types.jl")
include("constants.jl")
include("mice.jl")
include("classifyvariables.jl")
include("basefunctions.jl")
include("dataapifunctions.jl")
include("statsfunctions.jl")
include("glmfunctions.jl")
include("extras.jl")
include("testdataset.jl")

export mice
export describe, eltype, getvalues,
    betweenimputationvar, rubinsmean, rubinssterror, rubinsvar, 
    withinimputationsterrorsquared, withinimputationvar,
    componentmeans, componentstats, componentvars,
    mean, meanstats, std, summarystats, var,
    fit, glm, lm,
    desentinelize!, oneach,
    mcar, mcar!

end # module SimpleMice
