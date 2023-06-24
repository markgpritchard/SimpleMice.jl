
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Types for input data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MissInt = Union{Int, Missing}
MissBool = Union{Bool, Missing}
MissFloat = Union{T, Missing} where T <: Float64
MissNumber = Union{T, Missing} where T <: Number
#MissString = Union{T, Missing} where T <: AbstractString
MissString = Union{Missing, String}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Types for temporary dataset while imputing values
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

abstract type AbstractTempImputedValues end 

# Imputed values can be continuous or binary 
# (support for categorical variables will be added later)

abstract type ContinuousTempImputedValues <: AbstractTempImputedValues end 

abstract type BinaryTempImputedValues <: AbstractTempImputedValues end 

# Within continuous we have variables that were orignally integers, floats and potentially others 

mutable struct ContinuousIntTempImputedValues <: ContinuousTempImputedValues
    originalvalue       :: Int 
    originalmiss        :: Bool 
    imputedvalue        :: Float64 
end 

mutable struct ContinuousFloatTempImputedValues <: ContinuousTempImputedValues
    originalvalue       :: Float64 
    originalmiss        :: Bool 
    imputedvalue        :: Float64 
end 

mutable struct ContinuousAnyTempImputedValues <: ContinuousTempImputedValues
    originalvalue       :: Number 
    originalmiss        :: Bool 
    imputedvalue        :: Float64 
end 

# Within binary we have those that were originally Bool, integer, string or others

mutable struct BinaryBoolTempImputedValues <: BinaryTempImputedValues 
    originalvalue       :: Bool 
    originalmiss        :: Bool 
    probability         :: Float64 
    imputedvalue        :: Bool 
end 

mutable struct BinaryIntTempImputedValues <: BinaryTempImputedValues 
    originalvalue       :: Int 
    originalminimum     :: Int 
    originalmaximum     :: Int 
    originalmiss        :: Bool 
    probability         :: Float64 
    imputedvalue        :: Bool  
end 

mutable struct BinaryStringTempImputedValues <: BinaryTempImputedValues 
    originalvalue       :: AbstractString 
    originalminimum     :: AbstractString 
    originalmaximum     :: AbstractString 
    originalmiss        :: Bool 
    probability         :: Float64 
    imputedvalue        :: Bool  
end 

mutable struct BinaryAnyTempImputedValues <: BinaryTempImputedValues 
    originalvalue       :: Any 
    originalminimum     :: Any 
    originalmaximum     :: Any 
    originalmiss        :: Bool 
    probability         :: Float64 
    imputedvalue        :: Bool  
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Output type for imputed values
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

abstract type AbstractImputedData end 

abstract type ImputedVector <: AbstractImputedData end

struct ImputedDataFrame <: AbstractImputedData
    originaldf          :: DataFrame 
    numberimputed       :: Int 
    imputeddfs          :: Vector{DataFrame}
end 

struct ImputedVectorInt <: ImputedVector
    numberimputed       :: Int 
    imputedvalues       :: Vector{Vector{Int}}
end 

struct ImputedVectorFloat64 <: ImputedVector
    numberimputed       :: Int 
    imputedvalues       :: Vector{Vector{Float64}}
end 

struct ImputedVectorBool <: ImputedVector
    numberimputed       :: Int 
    imputedvalues       :: Vector{Vector{Bool}}
end 

struct ImputedVectorString <: ImputedVector
    numberimputed       :: Int 
    imputedvalues       :: Vector{Vector{<:AbstractString}}
end 

struct ImputedVectorAny <: ImputedVector
    numberimputed       :: Int 
    imputedvalues       :: Vector{Vector}
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Type to output results of regression of imputed datasets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# It would be nice to wrap the output of the regression of imputed datasets into 
# the same structure as the regression of a DataFrame, but for now this allows key 
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
