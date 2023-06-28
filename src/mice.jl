
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
    of binvars`, `contvars` and `noimputevars`. Default is `true` if vars is supplied 
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
    M = initializematrix(df, allvars, binvars, contvars, noimputevars)
    imputeddfs = [ impute!(M, allvars, binvars, contvars, noimputevars, df; 
        verbose, verbosei = i, kwargs...) 
        for i ∈ 1:n ]
    return ImputedDataFrame(df, n, imputeddfs)
    return M
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function initializematrix(df, allvars, binvars, contvars, noimputevars)
    numbervariables = length(allvars)
    M = Matrix{TempImputedValues}(undef, size(df, 1), numbervariables)
    initializematrix!(M, df, allvars, binvars, contvars, noimputevars)
    return M
end 

function initializematrix!(M, df, allvars, binvars, contvars, noimputevars)
    for (i, v) ∈ enumerate(allvars)
        if v ∈ binvars      M[:, i] = initializebinarytempvalues(df, v)     end 
        if v ∈ contvars     M[:, i] = initializecontinuoustempvalues(df, v) end 
        if v ∈ noimputevars M[:, i] = initializenoimputetempvalues(df, v)   end 
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
        properties.originalmax, value, value)
end 

function initializebinarytempvalue(value::T, properties::InitializeValues{T}, 
        originalmiss::Bool
    ) where T <: AbstractString 
    initialtruth = value == properties.originalmax
    return TempImputedValues{T}(value, ImputedBinary, originalmiss, properties.originalmin, 
    properties.originalmax, initialtruth, initialtruth)
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
        properties.originalmax, value, value)
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
        properties.originalmax, value, value)
end 

function initializenoimputetempvalue(value::T, properties::InitializeValues{T}) where T <: String
    initialtruth = value == properties.originalmax
    return TempImputedValues{T}(value, NoneImputed, false, properties.originalmin, 
        properties.originalmax, initialtruth, initialtruth)
end 

function _initializenoimputetempvalueswarning(properties::InitializeValues{T}) where T <: String
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
    values = getproperty(df, var)
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

function impute!(M, allvars, binvars, contvars, noimputevars, df; 
        initialvaluesfunc = sample, verbose, verbosei, kwargs...
    )
    if verbose @info "Starting imputation set $verbosei" end 
    initialvalues!(M, initialvaluesfunc)
    currentM, vec = currentmatrixandvector(M, 1)
    imputevalues!(currentM, vec, M, allvars, binvars, contvars; kwargs...)
    imputeddf = imputedf(df, M, allvars, noimputevars)
    return imputeddf
end 

function initialvalues!(M, initialvaluesfunc)
    for i ∈ axes(M, 2) 
        M[1, i].valuetype == NoneImputed && continue
        _initialvalues!(M[:, i], initialvaluesfunc) 
    end 
end 

_initialvalues!(v, initialvaluesfunc::Sample) = __initialvalues!(v, initialvaluesfunc)

# Counter - user receives exactly one notification about sending other functions 
# to initialvalues! with binary variables 
let initialbinvaluestate = 0
    global initialbinvaluecounter() = (initialbinvaluestate += 1)
end

function _initialvalues!(v, initialvaluesfunc)
    if v[i].valuetype == ImputedContinuous 
        __initialvalues!(v, initialvaluesfunc)
    else 
        if initialbinvaluecounter() == 1  
            @info """
            Initial values of binary variables are always selected by sample, regardless 
                of initialvaluesfunc argument
            """ 
        end 
        __initialvalues!(v, sample)
    end 
end 

function __initialvalues!(v, initialvaluesfunc) 
    nm = identifynonmissings(v)
    nmv = getproperty.(v[nm], :imputedvalue)
    for j ∈ eachindex(v) 
        if v[j].originalmiss 
            v[j].probability = initialvaluesfunc(nmv)
            v[j].imputedvalue = initialvaluesfunc(nmv)
        end
    end
end 

function imputevalues!(currentM, vec, M, allvars, binvars, contvars; m = 100)
    for _ ∈ 1:m _imputevalues!(currentM, vec, M, allvars, binvars, contvars) end 
end 

function _imputevalues!(currentM, vec, M, allvars, binvars, contvars)
    for (i, v) ∈ enumerate(allvars)
        if v ∈ binvars  imputebinvalues!(currentM, vec, M, i)  end 
        if v ∈ contvars imputecontvalues!(currentM, vec, M, i) end 
    end 
end 

function imputebinvalues!(currentM, vec, M, i) 
    for j ∈ eachindex(vec) vec[j] = currentM[j, i] end 
    currentM[:, i] = ones(size(currentM, 1))
    regr = fit(GeneralizedLinearModel, currentM, vec, Binomial())
    probabilities = predict(regr)
    for j ∈ axes(M, 1)
        if M[j, i].originalmiss
            M[j, i].probability = probabilities[j]
            M[j, i].imputedvalue = rand() < probabilities[j]
        end 
    end 
    currentM[:, i] = currentvalue.(M[:, i])
end 

function imputecontvalues!(currentM, v, M, i) 
    for j ∈ eachindex(v) v[1] = currentM[j, i] end 
    currentM[:, i] = ones(size(currentM, 1))
    regr = fit(LinearModel, currentM, v)
    predictions = predict(regr)
    for j ∈ axes(M, 1)
        if M[j, i].originalmiss
            M[j, i].imputedvalue = predictions[j]
        end 
    end 
    currentM[:, i] = currentvalue.(M[:, i])
end 

currentvalue(a) = a.imputedvalue

function currentmatrixandvector(M, i)
    currentM = currentvalue.(M)
    v = currentM[:, i]
    currentM[:, i] = ones(size(currentM, 1))
    return ( currentM, v ) 
end 

function imputedf(df, M, allvars, noimputevars)
    newdf = deepcopy(df)
    imputedf!(newdf, M, allvars, noimputevars)
    return newdf
end 

function imputedf!(newdf, M, allvars, noimputevars)
    for (i, var) ∈ enumerate(allvars) 
        var ∈ noimputevars && continue
        imputedfvector!(newdf, M[:, i], var) 
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
    return v.imputedvalue 
end 

function imputedvalue(v::TempImputedValues{T}) where T <: AbstractString 
    if v.imputedvalue == 1 return v.originalmaximum 
    else                   return v.originalminimum
    end
end 
