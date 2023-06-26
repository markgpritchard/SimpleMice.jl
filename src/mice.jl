
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
    if not supplied as positional arguments. If not supplied, variables with no 
    missing data classified as `noimputevars`, integers and strings with two unique 
    non-missing values, plus all `Bool` inputs, as `binvars`, and all other numbers 
    as `contvars`.
* `initialvaluesfunc = StatsBase.sample`: Function used to give each missing datapoint a value at the 
    start of the imputation algorithm. (Note that `sample` is used for all binary 
    [and categorical] variables regardless of this keyword argument.)
* `m = 100`: number of regressions to perform per variable with missing data during 
    the imputation process. [To do: make this more dynamic]
* `n = 5`: number of imputed datasets to produce.
* `printdropped`: whether to list members of vars that are not classified into one 
    of binvars`, `contvars` and `noimputevars`. Default is `true` if vars is supplied 
    and `false` otherwise.
* `verbose = true`: whether to display messages showing progress of the function.

## Examples 
To do
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

function mice(df, binvars::Vector, contvars::Vector, noimputevars::Vector; 
        n = 5, verbose = true, kwargs...
    ) 
    if verbose @info "Starting to initialize imputation process" end 
    tempdf = initializetempdf(df, binvars, contvars, noimputevars)
    imputeddfs = [ impute!(tempdf, binvars, contvars, noimputevars, df; 
        verbose, verbosei = i, kwargs...) 
        for i ∈ 1:n ]
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
    variable, nonmissings = _initializevalues(df, var)
    nmv = variable[nonmissings]
    originalmin = minimum(nmv)
    originalmax = maximum(nmv)
    return [ initializebinarytempvalue(variable[i], variable, nonmissings, originalmin, originalmax) 
        for i ∈ eachindex(variable) ]
end 

function initializebinarytempvalue(value::T, variable::Vector{<:Union{T, Missing}}, 
        nonmissings, originalmin::T, originalmax::T
    ) where T <: Number
    return BinaryTempImputedValues{T}(value, originalmin, originalmax, false, value, value)
end 

function initializebinarytempvalue(value::Missing, variable::Vector{<:Union{T, Missing}}, 
        nonmissings, originalmin::T, originalmax::T
    ) where T <: Number
    initialvalue = variable[sample(nonmissings)]
    return BinaryTempImputedValues{T}(initialvalue, originalmin, originalmax, true, initialvalue, initialvalue)
end 

function initializebinarytempvalue(value::T, variable::Vector{<:Union{T, Missing}}, 
        nonmissings, originalmin::T, originalmax::T
    ) where T <: AbstractString
    return initializebinarytempvaluestring(value, originalmin, originalmax, false)
end 

function initializebinarytempvalue(value::Missing, variable::Vector{<:Union{T, Missing}}, 
        nonmissings, originalmin::T, originalmax::T
    ) where T <: AbstractString
    initialvalue = variable[sample(nonmissings)]
    return initializebinarytempvaluestring(initialvalue, originalmin, originalmax, true)
end 

function initializebinarytempvaluestring(value::T, originalmin::T, originalmax::T, 
        originalmiss::Bool
    ) where T
    initialtruth = value == originalmax
    return BinaryTempImputedValues{T}(value, originalmin, originalmax, originalmiss, initialtruth, initialtruth)
end

function _initializevalues(df, var)
    variable = getproperty(df, var)
    nonmissings = identifynonmissings(variable)
    return ( variable, nonmissings )
end 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize continuous variables 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function initializecontinuoustempvalues(df::DataFrame, var::Symbol)
    variable, nonmissings = _initializevalues(df, var)
    return [ initializecontinuoustempvalue(variable[i], variable, nonmissings) 
        for i ∈ eachindex(variable) ]
end 

function initializecontinuoustempvalue(value::T, variable::Vector{<:Union{T, Missing}}, 
        nonmissings
    ) where T
    return ContinuousTempImputedValues{T}(value, false, value)
end 

function initializecontinuoustempvalue(value::Missing, variable::Vector{<:Union{T, Missing}}, 
        nonmissings
    ) where T 
    initialvalue = variable[sample(nonmissings)]
    return ContinuousTempImputedValues{T}(initialvalue, true, initialvalue)
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Identify non-missing values 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function identifynonmissings(variable::Vector{<:AbstractTempImputedValues}) 
    return findall(x -> !x.originalmiss, variable)
end

identifynonmissings(variable) = findall(x -> !ismissing(x), variable)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Impute values 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function impute!(tempdf, binvars, contvars, noimputevars, df; 
        m = 100, initialvaluesfunc = sample, verbose, verbosei, kwargs...
    )
    if verbose @info "Starting imputation set $verbosei" end 
    initialvalues!(tempdf, initialvaluesfunc, binvars, contvars, noimputevars)
    imputedvalues = imputevalues(tempdf, binvars, contvars; m)
    finaldf = preparefinaldf(tempdf, binvars, contvars, df) 
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

function insertinitialvalue!(tempdf, i, var, initialvalue::Number)
    tempdf[i, var].imputedvalue = initialvalue 
end 

function insertinitialvalue!(tempdf, i, var, initialvalue::AbstractString)
    initialtruth = initialvalue == tempdf[i, var].originalmaximum
    tempdf[i, var].imputedvalue = initialtruth 
end 

function makeworkingvector(df, var)
    variable = getproperty(df, var) 
    return makeworkingvector(variable)
end 

makeworkingvector(variable) = variable

function makeworkingvector(variable::Vector{<:AbstractTempImputedValues}) 
    return [ v.imputedvalue for v ∈ variable ]
end 

function makeworkingvector(variable::Vector{<:AbstractString}) 
    maxv = maximum(variable)
    return [ v == maxv for v ∈ variable ]
end 

function makeworkingvector(variable::Vector{ImputedVector{T}}) where T <: AbstractString
    maxv = maximum(variable.imputedvalues)
    return [ v.imputedvalue == maxv for v ∈ variable ]
end 

function makeworkingmatrix(df)
    M = ones(size(df))
    for (i, v) ∈ enumerate(Symbol.(names(df)))
        M[:, i] = makeworkingvector(df, v)
    end
    return M
end 

function imputevalues(tempdf, binvars, contvars; m)
    M = makeworkingmatrix(tempdf)
    binindices = findall(x -> x ∈ binvars, Symbol.(names(tempdf)))
    contindices = findall(x -> x ∈ contvars, Symbol.(names(tempdf)))
    for _ ∈ 1:m imputevalues!(M, binindices, contindices) end 
    imputedvalues = deepcopy(tempdf)
    for i ∈ binindices
        var = Symbol.(names(imputedvalues))[i]
        for (j, v) ∈ enumerate(getproperty(imputedvalues, var))
            if v.originalmiss 
                imputedvalues[j, var].probability = M[j, i]
                imputedvalues[j, var].imputedvalue = rand() < M[j, i]
            end # if v.originalmiss
        end # for (j, v) ∈ enumerate(getproperty(df, var))
    end # for i ∈ binindices
    for i ∈ contindices 
        var = Symbol.(names(imputedvalues))[i]
        for (j, v) ∈ enumerate(getproperty(imputedvalues, var)) 
            if v.originalmiss imputedvalues[j, var].imputedvalue = M[j, i] end 
        end # for (j, v) ∈ enumerate(getproperty(df, var)) 
    end # for i ∈ contindices 
end 

function imputevalues!(M, binindices, contindices) 
    for i ∈ axes(M, 2)
        if i ∈ binindices imputebinvalues!(M, i) end 
        if i ∈ contindices imputecontvalues!(M, i) end 
    end 
end 

function imputebinvalues!(M, i) 
    v = M[:, i]
    M[:, i] = ones(size(M, 1))
    regr = fit(GeneralizedLinearModel, M, v, Binomial())
    M[:, i] = predict(regr)
end 

function imputecontvalues!(M, i) 
    v = M[:, i]
    M[:, i] = ones(size(M, 1))
    regr = fit(LinearModel, M, v)
    M[:, i] = predict(regr)
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
    select!(newdf, Not(var))
    insertcols!(newdf, var => finaldfvalues(imputedvector))
end 

function finaldfvalues(imputedvector::Vector{ContinuousTempImputedValues{T}}) where T
    newvector::Vector{T} = [ finaldfvalue(v) for v ∈ imputedvector ] 
    return newvector 
end 

function finaldfvalues(imputedvector::Vector{BinaryTempImputedValues{T}}) where T
    newvector::Vector{T} = [ finaldfvalue(v) for v ∈ imputedvector ] 
    return newvector 
end 

finaldfvalue(v::ContinuousTempImputedValues) = v.imputedvalue 

function finaldfvalue(v::BinaryTempImputedValues)
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
