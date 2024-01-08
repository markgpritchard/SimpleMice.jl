
abstract type AbstractImputedData end 

struct ImputedNonMissingData{N, T} <: AbstractImputedData 
    v   :: T
end

struct ImputedMissingData{N, T} <: AbstractImputedData
    v   :: MVector{N, T}
end

ImputedData{N, T} = Union{ImputedNonMissingData{N, T}, ImputedMissingData{N, T}}

getvalue(a::ImputedNonMissingData, i) = a.v 
getvalue(a::ImputedMissingData, i) = a.v[i] 

#ImputedNonMissingData(v::T, N::Int) where T = ImputedNonMissingData{N, T}(v)
#ImputedMissingData(v::MVector{N, T}, N) where {N, T} = ImputedMissingData{N, T}(v)

#=

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Types for temporary dataset while imputing values
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@enum ImputedValueTypes ImputedBinary ImputedContinuous NoneImputed

struct VariableProperties
    variablename        :: Symbol 
    id                  :: Int
    variabletype        :: ImputedValueTypes 
    datatype            :: DataType
    originalmissings    :: Vector{Int}
    nmvec               :: Vector{Float64} 
    truestring          :: String 
    falsestring         :: String 
    truenumber          :: Float64 
    falsenumber         :: Float64
end 

struct InitialValues{T} 
    values              :: Vector{<:Union{Missing, T}}
    nonmissings         :: Vector{Int}
    nmv                 :: Vector{T}
    originalmin         :: T
    originalmax         :: T
end 

struct VariableCount
    binary              :: Int
    continuous          :: Int
    nonimputed          :: Int
    total               :: Int
end 

VariableCount(b, c, n) = VariableCount(b, c, n, b + c + n)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Output type for imputed values
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


struct ImputedDataFrame <: AbstractImputedData
    originaldf          :: DataFrame 
    numberimputed       :: Int 
    imputeddfs          :: Vector{DataFrame}
end 

struct ImputedVector{T} <: AbstractImputedData
    originalvector      :: Vector{<:Union{Missing, T}}
    numberimputed       :: Int 
    imputedvalues       :: Vector{Vector{T}} 
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Types to output results 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct ImputedResult{T} 
    results             :: Vector{T} 
    n                   :: Int 
end 

# It would be nice to wrap the output of the regression of imputed datasets into 
# the same structure as the regression of a DataFrame, but for now this allows main 
# results to be saved

struct ImputedRegressionResult
    formula             :: FormulaTerm
    n                   :: Int
    coefnames           :: Vector{String}
    coef                :: Vector{Float64}
    stderror            :: Vector{Float64}
    t                   :: Vector{Float64}
    pvalue              :: Vector{Float64}
    confint             :: Vector{Tuple{Float64, Float64}}
end
=#
