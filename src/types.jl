
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Types for temporary dataset while imputing values
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@enum ImputedValueTypes ImputedBinary ImputedContinuous NoneImputed
#=
abstract type ImputedValueTypes end 

struct ImputedBinary        <: ImputedValueTypes end
struct ImputedContinuous    <: ImputedValueTypes end
struct NoneImputed          <: ImputedValueTypes end
=#
#struct CurrentVector 
#    values              :: Vector{Float64} 
#end

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
  #  currentvalues       :: Vector{Float64} 
end 

#=
mutable struct TempImputedValues{T} where T <: ImputedValueTypes
    originalvalue       :: Float64 
    valuetype           :: T 
    datatype            :: DataType
    originalmiss        :: Bool 
    originalminimum     :: Float64 
    originalmaximum     :: Float64 
    probability         :: Float64 
    imputedvalue        :: Float64 
end 
=#
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
#=
struct VariableRange
    binary              :: UnitRange{Int} 
    continuous          :: UnitRange{Int} 
    nonimputed          :: UnitRange{Int} 
end 
=#

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Output type for imputed values
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

abstract type AbstractImputedData end 

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
# Type to output results of regression of imputed datasets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Type of function sample 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sample = typeof(sample)
