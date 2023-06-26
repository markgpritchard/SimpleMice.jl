
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Types for temporary dataset while imputing values
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

abstract type AbstractTempImputedValues end 

# Imputed values can be continuous or binary 
# (support for categorical variables is intended to be added later)

mutable struct ContinuousTempImputedValues{T} <: AbstractTempImputedValues 
    originalvalue       :: T 
    originalmiss        :: Bool 
    imputedvalue        :: Float64 
end 

mutable struct BinaryTempImputedValues{T} <: AbstractTempImputedValues 
    originalvalue       :: T 
    originalminimum     :: T 
    originalmaximum     :: T 
    originalmiss        :: Bool 
    probability         :: Float64 
    imputedvalue        :: Bool  
end 


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
