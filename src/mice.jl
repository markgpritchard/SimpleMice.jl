
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

Only binary and continuous variables are currently supported.

## Keyword arguments 
* Any of `binvars`, `contvars` and `noimputevars` can be supplied as keyword arguments 
    if not supplied as positional arguments. If not supplied, variables with no 
    missing data classified as `noimputevars`, integers and strings with two unique 
    non-missing values, plus all `Bool` inputs, as `binvars`, and all other numbers 
    as `contvars`.
* `initialvaluesfunc = StatsBase.sample`: Function used to give each missing datapoint 
    a value at the start of the imputation algorithm. (Note that `sample` is used 
    for all binary [and categorical] variables regardless of this keyword argument.)
* `m = 100`: number of regressions to perform per variable with missing data during 
    the imputation process. [To do: make this more dynamic]
* `n = 5`: number of imputed datasets to produce.
* `printdropped`: whether to list members of vars that are not classified into one 
    of `binvars`, `contvars` and `noimputevars`. Default is `true` if vars is supplied 
    and `false` otherwise.
* `verbose = true`: whether to display messages showing progress of the function.
"""
function mice(df; kwargs...)
    vars = names(df) # use all variable names 
    return mice(df, vars; printdropped = false, kwargs...)
end 

function mice(df, vars::Vector{<:AbstractString}; kwargs...)  
    symbolvars = symbollist(vars) # convert strings to symbols
    return mice(df, symbolvars; kwargs...)
end 

function mice(df, vars::Vector{Symbol}; 
        binvars = nothing, contvars = nothing, noimputevars = nothing, printdropped = true, 
        verbose = true, kwargs...
    ) 
    if verbose @info "Starting to classify variables" end 
    bv, cv, niv = classifyvars!(vars, binvars, contvars, noimputevars, df; printdropped)
    return mice(df, bv, cv, niv; verbose, kwargs...)
end 

function mice(df, binvars::Vector, contvars::Vector, noimputevars::Vector; kwargs...)
    allvars = [ binvars; contvars; noimputevars ]
    return _mice(df, allvars, binvars, contvars, noimputevars; kwargs...)
end

function mice(df, var1, vars...; kwargs...) 
    return mice(df, [ var1, vars... ]; kwargs...)
end  

function _mice(df, allvars, binvars, contvars, noimputevars; n = 5, verbose = true, kwargs...) 
    if verbose @info "Starting to initialize imputation process" end 
    variables = initializevariables(df, allvars, binvars, contvars, noimputevars)
    imputeddfs = [ impute!(variables, allvars, binvars, contvars, noimputevars, df; 
        verbose, verbosei = i, kwargs...) 
        for i ∈ 1:n ]
    return ImputedDataFrame(df, n, imputeddfs)
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function initializevariables(df, allvars::Vector{Symbol}, binvars, contvars, noimputevars)
    variables = Tuple([ initializevariables(df, var, binvars, contvars, noimputevars) 
        for var ∈ allvars ])
    return variables
end 

function initializevariables(df, var::Symbol, binvars, contvars, noimputevars)
    if     var ∈ binvars  return initializebinarytempvalues(df, var) 
    elseif var ∈ contvars return initializecontinuoustempvalues(df, var)
    else                  return initializenoimputetempvalues(df, var)
    end
end

function initializebinarytempvalues(df, var)
    properties = initializevalues(df, var)
    return initializebinarytempvalues(properties)
end 

function initializebinarytempvalues(properties)
    return [ initializebinarytempvalue(value, properties) for value ∈ properties.values ]
end 

function initializebinarytempvalue(value::T, properties::InitializeValues{T}) where T
    return initializebinarytempvalue(value, properties, false)
end 

function initializebinarytempvalue(value::Missing, properties::InitializeValues{T}) where T
    possiblevalue = sample(properties.nmv)
    return initializebinarytempvalue(possiblevalue, properties, true)
end 

function initializebinarytempvalue(value::T, properties::InitializeValues{T}, originalmiss) where T <: Number 
    return TempImputedValues{T}(value, ImputedBinary, originalmiss, properties.originalmin, 
        properties.originalmax, ImputedProbability(value), ImputedValue(value))
end 

function initializebinarytempvalue(value::T, properties::InitializeValues{T}, 
        originalmiss::Bool
    ) where T <: AbstractString 
    initialtruth = value == properties.originalmax
    return TempImputedValues{T}(value, ImputedBinary, originalmiss, properties.originalmin, 
        properties.originalmax, ImputedProbability(initialtruth), ImputedValue(initialtruth))
end

function initializecontinuoustempvalues(df, var)
    properties = initializevalues(df, var)
    return initializecontinuoustempvalues(properties)
end 

function initializecontinuoustempvalues(properties)
    return [ initializecontinuoustempvalue(value, properties) for value ∈ properties.values ]
end 

function initializecontinuoustempvalue(value::T, properties::InitializeValues{T}) where T
    return initializecontinuoustempvalue(value, properties, false)
end 

function initializecontinuoustempvalue(value::Missing, properties::InitializeValues{T}) where T
    possiblevalue = sample(properties.nmv)
    return initializecontinuoustempvalue(possiblevalue, properties, true)
end 

function initializecontinuoustempvalue(value::T, properties::InitializeValues{T}, originalmiss) where T
    return TempImputedValues{T}(value, ImputedContinuous, originalmiss, properties.originalmin, 
        properties.originalmax, ImputedProbability(value), ImputedValue(value))
end

function initializenoimputetempvalues(df, var)
    properties = initializevalues(df, var)
    return initializenoimputetempvalues(properties)
end 

function initializenoimputetempvalues(properties)
    _initializenoimputetempvalueswarning(properties)
    return [ initializenoimputetempvalue(value, properties) for value ∈ properties.values ]
end 

function initializenoimputetempvalue(value::T, properties::InitializeValues{T}) where T <: Number
    return TempImputedValues{T}(value, NoneImputed, false, properties.originalmin,
        properties.originalmax, ImputedProbability(value), ImputedValue(value))
end 

function initializenoimputetempvalue(value::T, properties::InitializeValues{T}) where T <: AbstractString
    initialtruth = value == properties.originalmax
    return TempImputedValues{T}(value, NoneImputed, false, properties.originalmin, 
        properties.originalmax, ImputedProbability(initialtruth), ImputedValue(initialtruth))
end 

function _initializenoimputetempvalueswarning(properties::InitializeValues{T}) where T <: AbstractString
    if length(unique(properties.nmv)) > 2 
        @warn """
        Categorical variables not currently supported, even for non-imputed values. 
        Function will continue but will give unreliable results.
        """
    end 
end 

_initializenoimputetempvalueswarning(properties) = nothing

function initializevalues(df, var)
    values = getproperty(df, var)
    return initializevalues(df, var, values)
end 

function initializevalues(df, var, values::Vector{<:Union{Missing, T}}) where T
    nonmissings = identifynonmissings(values)
    nmv = values[nonmissings]
    originalmin = minimum(nmv)
    originalmax = maximum(nmv)
    return InitializeValues{T}(values, nonmissings, nmv, originalmin, originalmax)
end 

function identifynonmissings(variable::Vector{TempImputedValues}) 
    return findall(x -> !x.originalmiss, variable)
end

identifynonmissings(variable) = findall(x -> !ismissing(x), variable)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Impute values 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function impute!(variables, allvars, binvars, contvars, noimputevars, df; 
        initialvaluesfunc = sample, verbose, verbosei, kwargs...
    )
    if verbose @info "Starting imputation set $verbosei" end 
    initialvalues!(variables, initialvaluesfunc)
    M = currentmatrix(variables)
    imputevalues!(M, variables, allvars, binvars, contvars; kwargs...)
    imputeddf = makeoutputdf(df, variables, allvars, noimputevars)
    return imputeddf
end 

function initialvalues!(variables, initialvaluesfunc)
    for i ∈ eachindex(variables)
        variables[i][1] == NoneImputed && continue
        _initialvalues!(variables[i], initialvaluesfunc) 
    end 
end 

_initialvalues!(vec, initialvaluesfunc::Sample) = __initialvalues!(vec, initialvaluesfunc)

# Counter - user receives exactly one notification about sending other functions 
# to initialvalues! with binary variables 
let initialbinvaluestate = 0
    global initialbinvaluecounter() = (initialbinvaluestate += 1)
end

function _initialvalues!(vec, initialvaluesfunc)
    if vec[1].valuetype == ImputedContinuous 
        __initialvalues!(vec, initialvaluesfunc)
    else 
        if initialbinvaluecounter() == 1  
            @info """
            Initial values of binary variables are always selected by sample, regardless 
                of initialvaluesfunc argument
            """ 
        end 
        __initialvalues!(vec, sample)
    end 
end 

function __initialvalues!(vec, initialvaluesfunc) 
    nm = identifynonmissings(vec)
   # _nmv = getproperty.(vec[nm], :imputedvalue)::Vector{ImputedValue}
    _nmv = [ getproperty(vec[i], :imputedvalue)::ImputedValue for i ∈ nm ]
    nmv = [ getproperty(val, :v)::Float64 for val ∈ _nmv ]
    __initialvalues!(vec, initialvaluesfunc, nmv) 
end 

function __initialvalues!(vec, initialvaluesfunc, nmv) 
    for j ∈ eachindex(vec) 
        if vec[j].originalmiss 
            vec[j].probability.p = initialvaluesfunc(nmv)
            vec[j].imputedvalue.v = initialvaluesfunc(nmv)
        end
    end
end 

function imputevalues!(M, variables, allvars, binvars, contvars; m = 100)
    for _ ∈ 1:m _imputevalues!(M, variables, allvars, binvars, contvars) end 
end 

function _imputevalues!(M, variables, allvars, binvars, contvars)
    for (i, var) ∈ enumerate(allvars)
        _imputevalues!(M, variables[i], binvars, contvars, i, var)
    end 
end 

function _imputevalues!(M, variable, binvars, contvars, i, var)
    if var ∈ binvars  
        imputebinvalues!(M, variable, i)  
    elseif var ∈ contvars 
        imputecontvalues!(M, variable, i) 
    end 
end 

function imputebinvalues!(M, variable, i::Int) 
    vec = deepcopy(M[:, i])
    M[:, i] = ones(size(M, 1))
    regr = fit(GeneralizedLinearModel, M, vec, Binomial())
    probabilities = predict(regr)
    imputebinvalues!(M, variable, probabilities) 
    M[:, i] = currentvalue.(variable)
end 

function imputebinvalues!(M, variable, probabilities::Vector{Float64}) 
    for j ∈ axes(M, 1)
        if variable[j].originalmiss
            variable[j].probability.p = probabilities[j]
            variable[j].imputedvalue.v = rand() < probabilities[j]
        end 
    end 
end 

function imputecontvalues!(M, variable, i::Int) 
    vec = deepcopy(M[:, i])
    M[:, i] = ones(size(M, 1))
    regr = fit(LinearModel, M, vec)
    predictions = predict(regr)
    imputecontvalues!(M, variable, predictions) 
    M[:, i] = currentvalue.(variable)
end 

function imputecontvalues!(M, variable, predictions::Vector{Float64}) 
    for j ∈ axes(M, 1)
        if variable[j].originalmiss
            variable[j].imputedvalue.v = predictions[j]
        end 
    end 
end 

currentvalue(a) = a.imputedvalue.v

function currentmatrix(variables)
    M = zeros(length(variables[1]), length(variables))
    for (i, vals) ∈ enumerate(variables) M[:, i] = currentvalue.(vals) end
    return M
end 

function makeoutputdf(df, variables, allvars, noimputevars)
    newdf = deepcopy(df)
    makeoutputdf!(newdf, variables, allvars, noimputevars)
    return newdf
end 

function makeoutputdf!(newdf, variables, allvars, noimputevars)
    for (i, var) ∈ enumerate(allvars) 
        var ∈ noimputevars && continue
        imputedfvector!(newdf, variables[i], var) 
    end
end 

function imputedfvector!(newdf, vector, var)
    select!(newdf, Not(var))
    insertcols!(newdf, var => imputedfvector(vector, vector[1]))
end 

function imputedfvector(vector, val::TempImputedValues{T}) where T
    newvector::Vector{T} = [ imputedvalue(v) for v ∈ vector ]
    return newvector 
end 

function imputedvalue(v::TempImputedValues{T}) where T <: Number 
    return v.imputedvalue.v 
end 

function imputedvalue(v::TempImputedValues{T}) where T <: AbstractString 
    if v.imputedvalue.v == 1 return v.originalmaximum 
    else                     return v.originalminimum
    end
end 
