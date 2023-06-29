
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
    # identify which variables will be treated as binary, continuous and non-missing
    bv, cv, niv = classifyvars!(vars, binvars, contvars, noimputevars, df; printdropped)
    return mice(df, bv, cv, niv; verbose, kwargs...)
end 

function mice(df, binvars::Vector, contvars::Vector, noimputevars::Vector; kwargs...)
    # move the function to
    return _mice(df, binvars, contvars, noimputevars; kwargs...)
end

function mice(df, var1, vars...; kwargs...) 
    return mice(df, [ var1, vars... ]; kwargs...)
end  

function _mice(df, binvars, contvars, noimputevars; n = 5, verbose = true, kwargs...) 
    if verbose @info "Starting to initialize imputation process" end 
    # count the variable types 
    variablecounts = VariableCount(length(binvars), length(contvars), length(noimputevars))
    tablelength = size(df, 1)
    variableproperties, M = getdetails(df, binvars, contvars, noimputevars, variablecounts, tablelength)
    vec = M[:, 1]
    imputeddfs = [ impute!(M, vec, variableproperties, df; verbose, verbosei = i, kwargs...) 
        for i ∈ 1:n ]
    return ImputedDataFrame(df, n, imputeddfs)
end 

function makevariableranges(vc)
    return VariableRange(
        UnitRange(1, vc.binary),
        UnitRange(vc.binary + 1, vc.binary + vc.continuous),
        UnitRange(vc.binary + vc.continuous + 1, vc.binary + vc.continuous + vc.nonimputed)
    )
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Variable properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function getdetails(df::DataFrame, binvars, contvars, noimputevars, variablecounts, tablelength) 
    M = zeros(tablelength, variablecounts.total) 
    variableproperties = (
        binarydict   = Dict([ String(var) => getdetailsbinary!(M, df, var, i) for (i, var) ∈ enumerate(binvars) ]),
        contdict     = Dict([ String(var) => getdetailscontinuous!(M, df, var, i + variablecounts.binary) for (i, var) ∈ enumerate(contvars) ]), 
        noimputedict = Dict([ String(var) => getdetailsnoimpute!(M, df, var, i + variablecounts.binary + variablecounts.continuous) for (i, var) ∈ enumerate(noimputevars) ]) 
    )
    return ( variableproperties, M )
end 

getdetailsbinary!(M, df, var, i) = getdetailsbincont!(M, df, var, i, ImputedBinary) 
getdetailscontinuous!(M, df, var, i) = getdetailsbincont!(M, df, var, i, ImputedContinuous) 

function getdetailsnoimpute!(M, df, var, i) 
    vec = getproperty(df, var)
    missings = Int[]
    return getdetails!(M, NoneImputed, df, var, vec, missings, i)
end 

function getdetailsbincont!(M, df, var, i, variabletype) 
    vec = getproperty(df, var)
    missings = identifymissings(vec)
    return getdetails!(M, variabletype, df, var, vec, missings, i)
end 

function getdetails!(M, variabletype, df, var, vec::Vector{<:Union{T, Missing}}, missings, i) where T <: AbstractString
    nmvec::Vector{T} = vec[Not(missings)]
    uniquevalues = unique(nmvec)
    @assert length(uniquevalues) == 2 "Function currently only supports binary values. Variable $var has $(length(uniquevalues)) unique values"
    maxvalue = maximum(uniquevalues)
    minvalue = minimum(uniquevalues)
    floatnmvec = [ v == maxvalue ? 1. : .0 for v ∈ nmvec ]
    M[:, i] = setinitialvalues(variabletype, vec, missings, nmvec, maxvalue, minvalue, floatnmvec)
    return VariableProperties(var, i, variabletype, T, missings, floatnmvec, maxvalue, minvalue, 0, 0)
end 

function getdetails!(M, variabletype, df, var, vec::Vector{<:Union{T, Missing}}, missings, i) where T <: Number
    nmvec::Vector{T} = vec[Not(missings)]
    maxvalue = maximum(nmvec)
    minvalue = minimum(nmvec)
    M[:, i] = setinitialvalues(variabletype, vec, missings, nmvec, maxvalue, minvalue)
    return VariableProperties(var, i, variabletype, T, missings, nmvec, "", "", maxvalue, minvalue)
end 

function setinitialvalues(variabletype, vec, missings, nmvec::Vector{T}, maxvalue, minvalue, floatnmvec) where T <: AbstractString
    currentvalues = zeros(length(vec))
    for i ∈ eachindex(vec)
        if i ∈ missings currentvalues[i] = Float64(sample(floatnmvec)) 
        else            currentvalues[i] = vec[i] == maxvalue ? 1. : .0
        end
    end 
    return currentvalues 
end 

function setinitialvalues(variabletype, vec, missings, nmvec::Vector{T}, maxvalue, minvalue) where T <: Number
    currentvalues = zeros(length(vec))
    for i ∈ eachindex(vec)
        if i ∈ missings currentvalues[i] = Float64(sample(nmvec)) 
        else            currentvalues[i] = Float64(vec[i])
        end
    end 
    return currentvalues 
end 

identifymissings(variable) = findall(x -> ismissing(x), variable)

identifynonmissings(variable) = findall(x -> !ismissing(x), variable)

function currentmatrix(variableproperties, variablecounts, tablelength)
    M = zeros(tablelength, variablecounts.total) 
    currentmatrix!(M, variableproperties)
    return M
end 

function currentmatrix!(M, variableproperties::NamedTuple)
    for d ∈ [ :binarydict, :contdict, :noimputedict ]
        currentmatrix!(M, getproperty(variableproperties, d))
    end
end 

function currentmatrix!(M, dict::Dict)
    for k ∈ keys(dict) M[:, dict[k].id] = dict[k].currentvalues end 
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Impute values 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function impute!(M, vec, variableproperties, df; 
        initialvaluesfunc = sample, verbose, verbosei, kwargs...
    )
    if verbose @info "Starting imputation set $verbosei" end 
    initialvalues!(M, variableproperties, initialvaluesfunc)
    imputevalues!(M, vec, variableproperties; kwargs...)
    imputeddf = makeoutputdf(df, variableproperties, M)
    return imputeddf
end 

function initialvalues!(M, variableproperties::NamedTuple, initialvaluesfunc)
    initialvalues!(M, variableproperties.binarydict, sample)
    initialvalues!(M, variableproperties.contdict, initialvaluesfunc)
end 

function initialvalues!(M, dict::Dict, initialvaluesfunc)
    for k ∈ keys(dict)
        initialvalues!(M, dict[k], initialvaluesfunc)
    end 
end 

function initialvalues!(M, variable::VariableProperties, initialvaluesfunc)
    i = variable.id
    initialvalues!(M, variable, initialvaluesfunc, i)
end 

function initialvalues!(M, variable, initialvaluesfunc, i)
    for j ∈ variable.originalmissings
        M[j, i] = initialvaluesfunc(variable.nmvec)
    end 
end 

function imputevalues!(M, vec, variableproperties; m = 100)
    for _ ∈ 1:m _imputevalues!(M, vec, variableproperties) end 
end 

function _imputevalues!(M, vec, variableproperties)
    imputevaluesbin!(M, vec, getproperty(variableproperties, :binarydict))
    imputevaluescont!(M, vec, getproperty(variableproperties, :contdict))
end

function imputevaluesbin!(M, vec, dict::Dict)
    for k ∈ keys(dict) imputevaluesbin!(M, vec, dict[k]) end 
end 

function imputevaluesbin!(M, vec, variable::VariableProperties)
    i = variable.id
    vec .= M[:, i]
    M[:, i] = ones(size(M, 1))
    regr = fit(GeneralizedLinearModel, M, vec, Binomial())
    probabilities = predict(regr)
    M[:, i] = vec
    imputevaluesbin!(M, variable, probabilities, i)
end 

function imputevaluesbin!(M, variable, probabilities, i)
    for j ∈ variable.originalmissings
        M[j, i] = rand() < probabilities[j] 
    end
end 

function imputevaluescont!(M, vec, dict::Dict)
    for k ∈ keys(dict) imputevaluescont!(M, vec, dict[k]) end 
end 

function imputevaluescont!(M, vec, variable::VariableProperties)
    i = variable.id
    vec .= M[:, i]
    M[:, i] = ones(size(M, 1))
    regr = fit(LinearModel, M, vec)
    predictions = predict(regr)
    M[:, i] = vec
    imputevaluescont!(M, variable, predictions, i)
end 

function imputevaluescont!(M, variable, predictions, i)
    for j ∈ variable.originalmissings
        M[j, i] = predictions[j] 
    end
end 

function makeoutputdf(df, variableproperties, M)
    newdf = deepcopy(df)
    makeoutputdf!(newdf, variableproperties, M)
    return newdf
end 

function makeoutputdf!(newdf, variableproperties::NamedTuple, M)
    for d ∈ [ :binarydict, :contdict ]
        makeoutputdf!(newdf, getproperty(variableproperties, d), M)
    end
end 

function makeoutputdf!(newdf, dict::Dict, M)
    for k ∈ keys(dict) makeoutputdf!(newdf, dict[k], M) end 
end

function makeoutputdf!(newdf, variable::VariableProperties, M)
    select!(newdf, Not(variable.variablename))
    insertcols!(newdf, variable.variablename => imputedfvector(variable, M))
end 

function imputedfvector(variable, M)
    datatype = variable.datatype
    return imputedfvector(datatype, variable, M)
end 

function imputedfvector(datatype, variable, M) 
    if datatype <: AbstractString return imputestringvector(datatype, variable, M) 
    else                          return imputenumbervector(datatype, variable, M) 
    end
end 

function imputenumbervector(datatype, variable, M)
    i = variable.id
    newvector::Vector{datatype} = M[:, i]
    return newvector
end 

function imputestringvector(datatype, variable, M)
    i = variable.id
    truestring = variable.truestring
    falsenumber = variable.falsestring
    newvector::Vector{datatype} = [ v == 1 ? truestring : falsenumber for v ∈ M[:, i] ]
    return newvector
end 
