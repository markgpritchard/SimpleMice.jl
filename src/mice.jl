
"""
    mice(df[, vars::Vector; <keyword arguments>]) 
    mice(df, var1, vars...[; <keyword arguments>]) 
    mice(df, bv::Vector, cv::Vector, niv::Vector[; <keyword arguments>]) 

Takes a `DataFrame` and imputes missing values by multiple imputation by chained equations.

`df` is the DataFrame with missing data. `vars` is a vector of variables to be used 
    for the multiple imputation algorithm. These can also be presented as arguments: 
    `var1, var2, var3...`. Variables can also be supplied in separate vectors, 
    `bv` for binary variables, `cv` for continuous variables, and `niv` for variables 
    with no missing data but which are to be used in the multiple imputation algorithm.
    If no variables are supplied to the function, all columns in the DataFrame will 
    be used.

## Keyword arguments 
To do 

## Examples 
To do
"""
function mice(df; kwargs...)
    vars = names(df) # use all variable names 
    return mice(df, vars; printdropped = false, kwargs...)
end 

function mice(df, vars::Vector{T}; kwargs...) where T <: AbstractString 
    symbolvars = symbollist(vars) # convert strings to symbols
    return mice(df, symbolvars; kwargs...)
end 

function mice(df, vars::Vector{T}; 
        binvars = nothing, contvars = nothing, noimputevars = nothing, printdropped = true, kwargs...
    ) where T <: Symbol 
    bv, cv, niv = classifyvars!(vars, binvars, contvars, noimputevars, df; printdropped)
    return mice(df, bv, cv, niv; kwargs...)
end 

function mice(df, bv::Vector, cv::Vector, niv::Vector; n = 5, kwargs...) 
    tempdf = initializetempdf(df, bv, cv, niv)
    imputeddfs = [ impute!(tempdf, bv, cv, niv, df; kwargs...) for _ ∈ 1:n ]
    return ImputedDataFrame(df, n, imputeddfs)
end 

function mice(df, var1, vars...; kwargs...) 
    return mice(df, [ var1, vars... ]; kwargs...)
end  


## Create a DataFrame that will hold temporary values for each variable 

function initializetempdf(df, binvars, contvars, noimputevars; kwargs...)
    tempdf = DataFrame(
        [ var => initializebinarytempvalues(df, var) for var ∈ binvars ];
        [ var => initializecontinuoustempvalues(df, var) for var ∈ contvars ];
        [ var => getproperty(df, var) for var ∈ noimputevars ]
    )
    return tempdf
end 

## Initialize binary variables 

function initializebinarytempvalues(df::DataFrame, var::Symbol; kwargs...)
    variable = getproperty(df, var)
    nonmissings = identifynonmissings(variable)
    return initializebinarytempvalues(variable, nonmissings)
end 

function initializebinarytempvalues(variable::Vector{T}, nonmissings::Vector; kwargs...) where T <: MissBool
    return [ initializebinarytempvalue(variable[i], variable, nonmissings) for i ∈ eachindex(variable) ]
end 

function initializebinarytempvalues(variable::Vector, nonmissings::Vector; kwargs...) 
    nmv = variable[nonmissings]
    originalmin = minimum(nmv)
    originalmax = maximum(nmv)
    return [ initializebinarytempvalue(variable[i], variable, nonmissings, originalmin, originalmax) for i ∈ eachindex(variable) ]
end 

function initializebinarytempvalue(value::Bool, variable::Vector, nonmissings)
    return BinaryBoolTempImputedValues(value, false, value, value)
end 

function initializebinarytempvalue(value::Missing, variable::Vector, nonmissings)
    initialvalue = variable[sample(nonmissings)]
    return BinaryBoolTempImputedValues(initialvalue, true, initialvalue, initialvalue)
end 

function initializebinarytempvalue(value::Int, variable::Vector{T}, nonmissings, originalmin, originalmax) where T <: MissInt
    return BinaryIntTempImputedValues(value, originalmin, originalmax, false, value, value)
end 

function initializebinarytempvalue(value::Missing, variable::Vector{T}, nonmissings, originalmin, originalmax) where T <: MissInt
    initialvalue = variable[sample(nonmissings)]
    return BinaryIntTempImputedValues(initialvalue, originalmin, originalmax, true, initialvalue, initialvalue)
end 

function initializebinarytempvalue(value::AbstractString, variable::Vector{T}, nonmissings, originalmin, originalmax) where T <: MissString
    return BinaryStringTempImputedValues(value, originalmin, originalmax, false, value, value)
end 

function initializebinarytempvalue(value::Missing, variable::Vector{T}, nonmissings, originalmin, originalmax) where T <: MissString
    initialvalue = variable[sample(nonmissings)]
    return BinaryStringTempImputedValues(initialvalue, originalmin, originalmax, true, initialvalue, initialvalue)
end 

function initializebinarytempvalue(value, variable::Vector, nonmissings, originalmin, originalmax)
    return BinaryAnyTempImputedValues(value, originalmin, originalmax, false, value, value)
end 

function initializebinarytempvalue(value::Missing, variable::Vector, nonmissings, originalmin, originalmax)
    initialvalue = variable[sample(nonmissings)]
    return BinaryAnyTempImputedValues(initialvalue, originalmin, originalmax, true, initialvalue, initialvalue)
end 


## Initialize continuous variables 

function initializecontinuoustempvalues(df::DataFrame, var::Symbol; kwargs...)
    variable = getproperty(df, var)
    nonmissings = identifynonmissings(variable)
    return initializecontinuoustempvalues(variable, nonmissings)
end 

function initializecontinuoustempvalues(variable::Vector, nonmissings::Vector; kwargs...) 
    return [ initializecontinuoustempvalue(variable[i], variable, nonmissings) for i ∈ eachindex(variable) ]
end 

function initializecontinuousinttempvalue(value::Int, variable::Vector{T}, nonmissings) where T <: MissInt
    return ContinuousIntTempImputedValues(value, false, value)
end 

function initializecontinuousinttempvalue(value::Missing, variable::Vector{T}, nonmissings) where T <: MissInt
    initialvalue = variable[sample(nonmissings)]
    return ContinuousIntTempImputedValues(initialvalue, true, initialvalue)
end 

function initializecontinuousinttempvalue(value::Float64, variable::Vector{T}, nonmissings) where T <: MissFloat
    return ContinuousFloatTempImputedValues(value, false, value)
end 

function initializecontinuousinttempvalue(value::Missing, variable::Vector{T}, nonmissings) where T <: MissFloat
    initialvalue = variable[sample(nonmissings)]
    return ContinuousFloatTempImputedValues(initialvalue, true, initialvalue)
end 

function initializecontinuousinttempvalue(value::Number, variable::Vector{T}, nonmissings) where T <: MissNumber
    return ContinuousAnyTempImputedValues(value, false, value)
end 

function initializecontinuousinttempvalue(value::Missing, variable::Vector{T}, nonmissings) where T <: MissNumber
    initialvalue = variable[sample(nonmissings)]
    return ContinuousAnyTempImputedValues(initialvalue, true, initialvalue)
end 


## Identify non-missing values 

function identifynonmissings(variable::Vector{T}) where T <: AbstractTempImputedValues
    return findall(x -> !x.originalmiss, variable)
end

identifynonmissings(variable) = findall(x -> !ismissing(x), variable)


## Impute values 

function impute!(tempdf, binvars, contvars, noimputevars, df; m = 100, initialvaluesfunc = sample, kwargs...)
    initialvalues!(tempdf, initialvaluesfunc, binvars, contvars, noimputevars)
    for _ ∈ 1:m imputevalues!(tempdf, binvars, contvars; kwargs...) end 
    finaldf = preparefinaldf(tempdf, binvars, contvars, df) 
    return finaldf
end 

function initialvalues!(df, initialvaluesfunc, binvars, contvars, noimputevars)
    for v ∈ binvars initialbinvalue!(df, initialvaluesfunc, v) end 
    for v ∈ contvars initialvalue!(df, initialvaluesfunc, v) end 
end

initialbinvalue!(df, initialvaluesfunc::Sample, v) = initialvalue!(df, initialvaluesfunc, v)

# Counter - user receives exactly one notification about sending other functions to initialbinvalue!
let initialbinvaluestate = 0
    global initialbinvaluecounter() = (initialbinvaluestate += 1)
end

function initialbinvalue!(df, initialvaluesfunc, v) 
    if initialbinvaluecounter() == 1  
        @info """
        Initial values of binary variables are always selected by sample, regardless of initialvaluesfunc argument
        """ 
    end 
    initialvalue!(df, sample, v)
end 

function initialvalue!(df, initialvaluesfunc, var)
    variable = getproperty(df, var)
    nm = identifynonmissings(variable)
    nmvariablevalues = [ variable[i].originalvalue for i ∈ nm ]
    for (i, v) ∈ enumerate(variable)
        if variable.originalmiss 
            initialvalue = initialvaluesfunc(nmvariablevalues)
            df[i, var].imputedvalue = initialvalue 
        end 
    end 
end 

function makeworkingdf(df) 
    workingdf = DataFrame(
        [ var => makeworkingdf(df, var) for var ∈ binvars ];
        [ var => makeworkingdf(df, var) for var ∈ contvars ];
        [ var => getproperty(df, var) for var ∈ noimputevars ]
    )
    return workingdf
end 

function makeworkingdf(df, var)
    variable = getproperty(df, var) 
    return [ v.imputedvalue for v ∈ variable ]
end 

function imputevalues!(df, binvars, contvars; kwargs...) 
    for var ∈ binvars imputebinvalues!(df, var; kwargs...) end
    for var ∈ contvars imputecontvalues!(df, var; kwargs...) end
end 

function imputebinvalues!(df, var; kwargs...) 
    workingdf, formula = prepareimputevalues(df, var; kwargs...) 
    regr = glm(formula, df, Binomial(), ProbitLink())
    predictions = predict(regr)
    for (i, v) ∈ enumerate(getproperty(df, var)) 
        if v.originalmiss 
            df[i, var].probability = predictions[i]
            df[i, var].imputedvalue = rand < predictions[i]
        end 
    end 
end 

function imputecontvalues!(df, var; kwargs...) 
    workingdf, formula = prepareimputevalues(df, var; kwargs...) 
    regr = lm(formula, df)
    predictions = predict(regr)
    for (i, v) ∈ enumerate(getproperty(df, var)) 
        if v.originalmiss df[i, var].imputedvalue = predictions[i] end 
    end 
end 

function prepareimputevalues(df, var; kwargs...) 
    workingdf = makeworkingdf(df)
    formula = Term(var) ~ sum(Term.(Symbol.(names(df[:, Not(var)]))))
    return ( workingdf, formula )
end 

function preparefinaldf(tempdf, binvars, contvars, df) 
    newdf = deepcopy(df) 
    for var ∈ names(newdf) 
        if var ∈ binvars || var ∈ contvars
            newdf[:, var] = [ tempdf[i, var].imputedvalue for i ∈ eachindex(getproperty(tempdf, var)) ]
        end 
    end 
    return newdf
end 