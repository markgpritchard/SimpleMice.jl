
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Types for input data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MissInt = Union{Int, Missing}
MissBool = Union{Bool, Missing}
MissFloat = Union{T, Missing} where T <: Float64
MissNumber = Union{T, Missing} where T <: Number
MissString = Union{T, Missing} where T <: AbstractString


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
    orginalmiss         :: Bool 
    probability         :: Float64 
    imputedvalue        :: Bool  
end 

mutable struct BinaryStringTempImputedValues <: BinaryTempImputedValues 
    originalvalue       :: AbstractString 
    originalminimum     :: AbstractString 
    originalmaximum     :: AbstractString 
    orginalmiss         :: Bool 
    probability         :: Float64 
    imputedvalue        :: Bool  
end 

mutable struct BinaryAnyTempImputedValues <: BinaryTempImputedValues 
    originalvalue       :: Any 
    originalminimum     :: Any 
    originalmaximum     :: Any 
    orginalmiss         :: Bool 
    probability         :: Float64 
    imputedvalue        :: Bool  
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Output type for imputed values
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct ImputedDataFrame 
    originaldf          :: DataFrame 
    numberimputed       :: Int 
    imputeddfs          :: Vector{DataFrame}
end 

struct ImputedVectorInt
    numberimputed       :: Int 
    imputedvalues       :: Vector{Int}
end 

struct ImputedVectorFloat64
    numberimputed       :: Int 
    imputedvalues       :: Vector{Float64}
end 

struct ImputedVectorBool
    numberimputed       :: Int 
    imputedvalues       :: Vector{Bool}
end 

struct ImputedVectorString
    numberimputed       :: Int 
    imputedvalues       :: Vector{<:AbstractString}
end 

struct ImputedVectorAny
    numberimputed       :: Int 
    imputedvalues       :: Vector
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Type of function sample 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sample = typeof(sample)
