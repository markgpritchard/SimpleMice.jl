
"""
    mice(df[, vars::Vector]; <keyword arguments>) 
    mice(df, var1, vars...; <keyword arguments>) 
    mice(df, binvars::Vector, contvars::Vector, noimputevars::Vector; <keyword arguments>) 

Takes a `DataFrame` and imputes missing values using multiple imputation by chained 
    equations.

`df` is the DataFrame with missing data. `vars` is a vector of variables to be used 
    for the multiple imputation algorithm. These can also be presented as arguments: 
    `var1, var2, var3...`. Variables can also be supplied in separate vectors, 
    `binvars` for binary variables, `contvars` for continuous variables, and `noimputevars` 
    for variables with no missing data but which are to be used in the multiple 
    imputation algorithm. If no variables are supplied to the function, all columns 
    in the DataFrame will be used.

## Keyword arguments 
* Any of `binvars`, `contvars` and `noimputevars` can be supplied as keyword arguments 
    if not supplied as positional arguments.
* `initialvaluesfunc`: Function used to give each missing datapoint a value at the 
    start of the imputation algorithm. Defult = `StatsBase.sample`. (Note that `sample` 
    is used for all binary [and categorical] variables regardless of this keyword 
    argument.)
* `m`: number of regressions to perform per variable with missing data during the 
    imputation process. Default = 100. [To do: make this more dynamic]
* `n`: number of imputed datasets to produce. Default = 5.

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

function mice(df, binvars::Vector, contvars::Vector, noimputevars::Vector; n = 5, kwargs...) 
    tempdf = initializetempdf(df, binvars, contvars, noimputevars)
    imputeddfs = [ impute!(tempdf, binvars, contvars, noimputevars, df; kwargs...) 
        for _ ∈ 1:n ]
    return ImputedDataFrame(df, n, imputeddfs)
end 

function mice(df, var1, vars...; kwargs...) 
    return mice(df, [ var1, vars... ]; kwargs...)
end  


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a DataFrame that will hold temporary values for each variable 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function initializetempdf(df, binvars, contvars, noimputevars)
    tempdf = DataFrame([
        [ var => initializebinarytempvalues(df, var) for var ∈ binvars ];
        [ var => initializecontinuoustempvalues(df, var) for var ∈ contvars ];
        [ var => getproperty(df, var) for var ∈ noimputevars ]
    ])
    return tempdf
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize binary variables 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function initializebinarytempvalues(df::DataFrame, var::Symbol)
    variable = getproperty(df, var)
    nonmissings = identifynonmissings(variable)
    return initializebinarytempvalues(variable, nonmissings)
end 

function initializebinarytempvalues(variable::Vector{T}, nonmissings::Vector) where T <: MissBool
    return [ initializebinarytempvalue(variable[i], variable, nonmissings) 
        for i ∈ eachindex(variable) ]
end 

function initializebinarytempvalues(variable::Vector, nonmissings::Vector) 
    nmv = variable[nonmissings]
    originalmin = minimum(nmv)
    originalmax = maximum(nmv)
    return [ initializebinarytempvalue(variable[i], variable, nonmissings, originalmin, originalmax) 
        for i ∈ eachindex(variable) ]
end 

function initializebinarytempvalue(value::Bool, variable::Vector, nonmissings)
    return BinaryBoolTempImputedValues(value, false, value, value)
end 

function initializebinarytempvalue(value::Missing, variable::Vector, nonmissings)
    initialvalue = variable[sample(nonmissings)]
    return BinaryBoolTempImputedValues(initialvalue, true, initialvalue, initialvalue)
end 

function initializebinarytempvalue(value::Int, variable::Vector{T}, nonmissings, 
        originalmin, originalmax
    ) where T <: MissInt
    return BinaryIntTempImputedValues(value, originalmin, originalmax, false, value, value)
end 

function initializebinarytempvalue(value::Missing, variable::Vector{T}, nonmissings, 
        originalmin, originalmax
    ) where T <: MissInt
    initialvalue = variable[sample(nonmissings)]
    return BinaryIntTempImputedValues(initialvalue, originalmin, originalmax, true, 
        initialvalue, initialvalue)
end 

function initializebinarytempvalue(value::AbstractString, variable::Vector{T}, 
        nonmissings, originalmin, originalmax
    ) where T <: MissString
    originalmiss = false
    return initializebinarytempvalue(value, variable, nonmissings, originalmin, 
        originalmax, originalmiss, value) 
end 

function initializebinarytempvalue(value::Missing, variable::Vector{T}, nonmissings, 
        originalmin, originalmax
    ) where T <: MissString
    originalmiss = true
    initialvalue = variable[sample(nonmissings)]
    return initializebinarytempvalue(value, variable, nonmissings, originalmin,
        originalmax, originalmiss, initialvalue) 
end 

function initializebinarytempvalue(value, variable::Vector{T}, nonmissings, originalmin, 
        originalmax, originalmiss, initialvalue
    ) where T <: MissString
    initialtruth = initialvalue == originalmax
    return BinaryStringTempImputedValues(initialvalue, originalmin, originalmax, 
        originalmiss, initialtruth, initialtruth)
end 

function initializebinarytempvalue(value, variable::Vector, nonmissings, originalmin, originalmax)
    return BinaryAnyTempImputedValues(value, originalmin, originalmax, false, value, value)
end 

function initializebinarytempvalue(value::Missing, variable::Vector, nonmissings, 
        originalmin, originalmax
    )
    initialvalue = variable[sample(nonmissings)]
    return BinaryAnyTempImputedValues(initialvalue, originalmin, originalmax, true, 
        initialvalue, initialvalue)
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize continuous variables 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function initializecontinuoustempvalues(df::DataFrame, var::Symbol)
    variable = getproperty(df, var)
    nonmissings = identifynonmissings(variable)
    return initializecontinuoustempvalues(variable, nonmissings)
end 

function initializecontinuoustempvalues(variable::Vector, nonmissings::Vector) 
    return [ initializecontinuoustempvalue(variable[i], variable, nonmissings) 
        for i ∈ eachindex(variable) ]
end 

function initializecontinuoustempvalue(value::Int, variable::Vector{T}, nonmissings) where T <: MissInt
    return ContinuousIntTempImputedValues(value, false, value)
end 

function initializecontinuoustempvalue(value::Missing, variable::Vector{T}, nonmissings) where T <: MissInt
    initialvalue = variable[sample(nonmissings)]
    return ContinuousIntTempImputedValues(initialvalue, true, initialvalue)
end 

function initializecontinuoustempvalue(value::Float64, variable::Vector{T}, nonmissings) where T <: MissFloat
    return ContinuousFloatTempImputedValues(value, false, value)
end 

function initializecontinuoustempvalue(value::Missing, variable::Vector{T}, nonmissings) where T <: MissFloat
    initialvalue = variable[sample(nonmissings)]
    return ContinuousFloatTempImputedValues(initialvalue, true, initialvalue)
end 

function initializecontinuoustempvalue(value::Number, variable::Vector{T}, nonmissings) where T <: MissNumber
    return ContinuousAnyTempImputedValues(value, false, value)
end 

function initializecontinuoustempvalue(value::Missing, variable::Vector{T}, nonmissings) where T <: MissNumber
    initialvalue = variable[sample(nonmissings)]
    return ContinuousAnyTempImputedValues(initialvalue, true, initialvalue)
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Identify non-missing values 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function identifynonmissings(variable::Vector{T}) where T <: AbstractTempImputedValues
    return findall(x -> !x.originalmiss, variable)
end

identifynonmissings(variable) = findall(x -> !ismissing(x), variable)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Impute values 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function impute!(tempdf, binvars, contvars, noimputevars, df; 
        m = 100, initialvaluesfunc = sample, kwargs...
    )
    td = deepcopy(tempdf)
    initialvalues!(td, initialvaluesfunc, binvars, contvars, noimputevars)
    for _ ∈ 1:m imputevalues!(td, binvars, contvars; kwargs...) end 
    finaldf = preparefinaldf(td, binvars, contvars, df) 
    return finaldf
end 

function initialvalues!(tempdf, initialvaluesfunc, binvars, contvars, noimputevars)
    for v ∈ binvars initialbinvalue!(tempdf, initialvaluesfunc, v) end 
    for v ∈ contvars initialvalue!(tempdf, initialvaluesfunc, v) end 
end

initialbinvalue!(tempdf, initialvaluesfunc::Sample, v) = initialvalue!(tempdf, initialvaluesfunc, v)

# Counter - user receives exactly one notification about sending other functions to initialbinvalue!
let initialbinvaluestate = 0
    global initialbinvaluecounter() = (initialbinvaluestate += 1)
end

function initialbinvalue!(tempdf, initialvaluesfunc, v) 
    if initialbinvaluecounter() == 1  
        @info """
        Initial values of binary variables are always selected by sample, regardless 
            of initialvaluesfunc argument
        """ 
    end 
    initialvalue!(tempdf, sample, v)
end 

function initialvalue!(tempdf, initialvaluesfunc, var)
    variable = getproperty(tempdf, var)
    nm = identifynonmissings(variable)
    nmvariablevalues = [ variable[i].originalvalue for i ∈ nm ]
    for (i, v) ∈ enumerate(variable)
        if v.originalmiss 
            initialvalue = initialvaluesfunc(nmvariablevalues)
            insertinitialvalue!(tempdf, i, var, initialvalue)
        end 
    end 
end 

function insertinitialvalue!(tempdf, i, var, initialvalue)
    tempdf[i, var].imputedvalue = initialvalue 
end 

function insertinitialvalue!(tempdf, i, var, initialvalue::AbstractString)
    initialtruth = initialvalue == tempdf[i, var].originalmaximum
    tempdf[i, var].imputedvalue = initialtruth 
end 

makeworkingdf(df) = DataFrame([ var => makeworkingdf(df, var) for var ∈ names(df) ])

function makeworkingdf(df, var)
    variable = getproperty(df, var) 
    return _makeworkingdf(variable)
end 

_makeworkingdf(variable) = variable

function _makeworkingdf(variable::Vector{T}) where T <: AbstractTempImputedValues
    return [ v.imputedvalue for v ∈ variable ]
end 

function imputevalues!(df, binvars, contvars; kwargs...) 
    for var ∈ binvars imputebinvalues!(df, var; kwargs...) end
    for var ∈ contvars imputecontvalues!(df, var; kwargs...) end
end 

function imputebinvalues!(df, var; kwargs...) 
    workingdf, fla = prepareimputevalues(df, var; kwargs...) 
    regr = fit(GeneralizedLinearModel, fla, workingdf, Binomial())
    predictions = predict(regr)
    for (i, v) ∈ enumerate(getproperty(df, var)) 
        if v.originalmiss 
            df[i, var].probability = predictions[i]
            df[i, var].imputedvalue = rand() < predictions[i]
        end 
    end 
end 

function imputecontvalues!(df, var; kwargs...) 
    workingdf, fla = prepareimputevalues(df, var; kwargs...) 
    regr = fit(LinearModel, fla, workingdf)
    predictions = predict(regr)
    for (i, v) ∈ enumerate(getproperty(df, var)) 
        if v.originalmiss df[i, var].imputedvalue = predictions[i] end 
    end 
end 

function prepareimputevalues(df, var; kwargs...) 
    workingdf = makeworkingdf(df)
    fla = Term(var) ~ sum(Term.(Symbol.(names(df[:, Not(var)]))))
    return ( workingdf, fla )
end 

function preparefinaldf(td, binvars, contvars, df) 
    newdf = deepcopy(df) 
    for var ∈ Symbol.(names(newdf))
        if var ∈ binvars || var ∈ contvars
            imputedvector = getproperty(td, var) 
            updatedfinaldf!(newdf, imputedvector, var)
        end 
    end 
    return newdf
end 

function updatedfinaldf!(newdf, imputedvector, var) 
    newdf[:, var] = [ finaldfvalue(v) for v ∈ imputedvector ] 
end 

finaldfvalue(v) = v.imputedvalue 

function finaldfvalue(v::BinaryStringTempImputedValues)
    if v.originalmiss 
        if v.imputedvalue 
            return v.originalmaximum 
        else # not v.imputedvalue 
            return v.originalminimum 
        end # if v.imputedvalue 
    else # not v.originalmiss
        return v.originalvalue 
    end # if v.originalmiss
end 
