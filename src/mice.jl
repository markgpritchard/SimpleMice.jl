
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
* `printdropped`: whether to list members of `vars` that are not classified into one 
    of `binvars`, `contvars` and `noimputevars`. Default is `true` if `vars` is supplied 
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

function mice(df, binvars::Vector, contvars::Vector, noimputevars::Vector; formulas = nothing, kwargs...)
    return _mice(df, binvars, contvars, noimputevars, formulas; kwargs...)
end

function mice(df, var1, vars...; kwargs...) 
    return mice(df, [ var1, vars... ]; kwargs...)
end  

function _mice(df, binvars, contvars, noimputevars, formulas::Nothing; verbose = true, kwargs...) 
    # When formula is nothing (i.e. default of linear regression with respect to 
    # all other variables) can use more efficient code that does all regressions 
    # using a matrix and vector that are mutated to reduce memory allocations
    if verbose @info "Starting to initialize imputation process" end 
    # count the variable types and the size of df to set the size of the matrix used 
    # in the regressions
    variablecounts = VariableCount(length(binvars), length(contvars), length(noimputevars))
    tablelength = size(df, 1)
    # variableproperties is a tuple of Dicts of details of each variable being used 
    # in the imputation. This is not mutated during the imputation but is passed 
    # around to guide the imputation.
    variableproperties, M = getdetails_m(df, binvars, contvars, noimputevars, variablecounts, tablelength)
    vec = deepcopy(M[:, 1])
    # M and vec are respectively a matrix of independent variables and a vector of 
    # dependent variables used in each regression. They are introduced here so that 
    # subsequent functions can be mutating rather than creating multiple matrices 
    # and vectors, which would require a lot of memory allocation. The values in 
    # them at this stage are unimportant. Note that M and vec each contain "Float64" 
    # values. Details in `variableproperties` allow these to be converted back to 
    # their original data types. However, this currently prevents use of categorical 
    # variables except binary variables.
    return _mice(M, vec, variableproperties, df; verbose, kwargs...) 
end 

function _mice(df, binvars, contvars, noimputevars, formulas::Vector{<:FormulaTerm}; verbose = true, kwargs...) 
    @assert length(formulas) == length(binvars) + length(contvars)
    if verbose @info "Starting to initialize imputation process" end 
    # count the variable types and the size of df to set the size of the matrix used 
    # in the regressions
    variablecounts = VariableCount(length(binvars), length(contvars), length(noimputevars))
    tablelength = size(df, 1)
    # variableproperties is a tuple of Dicts of details of each variable being used 
    # in the imputation. This is not mutated during the imputation but is passed 
    # around to guide the imputation.
    variableproperties, tdf = getdetails_tdf(df, binvars, contvars, noimputevars, variablecounts, tablelength)
    vec = formulas
    return _mice(tdf, vec, variableproperties, df; verbose, kwargs...) 
end 

function _mice(M_t, vec, variableproperties, df; n = 5, verbose, kwargs...) 
    imputeddfs = Vector{DataFrame}(undef, n)
    return mice!(imputeddfs, M_t, vec, variableproperties, df; n, verbose, kwargs...) 
end 

function mice!(imputeddfs, M_t, vec, variableproperties, df; n, verbose, kwargs...)
    Threads.@threads for i ∈ 1:n 
        imputeddfs[i] = impute(M_t, vec, variableproperties, df; verbose, verbosei = i, kwargs...)  
    end 
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

# In the following code, M is a Matrix of values used for regression calculations, 
# tdf is a temporary DataFrame used for regression calculations, and M_t is used 
# in code that can accept either M or tdf

function getdetails_m(df, binvars, contvars, noimputevars, variablecounts, tablelength) 
    M = zeros(tablelength, variablecounts.total) 
    return getdetails!(M, df, binvars, contvars, noimputevars, variablecounts, tablelength) 
end

function getdetails_tdf(df, binvars, contvars, noimputevars, variablecounts, tablelength) 
    tdf = DataFrame()
    return getdetails!(tdf, df, binvars, contvars, noimputevars, variablecounts, tablelength) 
end

function getdetails(df, binvars, contvars, noimputevars, variablecounts, tablelength) 
    return getdetails_m(df, binvars, contvars, noimputevars, variablecounts, tablelength)  
end

function getdetails!(M_t, df, binvars, contvars, noimputevars, variablecounts, tablelength) 
    variableproperties = (
        binarydict   = Dict(
            [ String(var) => getdetailsbinary!(M_t, df, var, i) for (i, var) ∈ enumerate(binvars) ]
        ),
        contdict     = Dict(
            [ String(var) => getdetailscontinuous!(M_t, df, var, i + variablecounts.binary) 
                for (i, var) ∈ enumerate(contvars) ]
        ), 
        noimputedict = Dict(
            [ String(var) => getdetailsnoimpute!(M_t, df, var, i + variablecounts.binary + variablecounts.continuous) 
                for (i, var) ∈ enumerate(noimputevars) ]
        ) 
    )
    return ( variableproperties, M_t )
end 

getdetailsbinary!(M_t, df, var, i) = getdetailsbincont!(M_t, df, var, i, ImputedBinary) 
getdetailscontinuous!(M_t, df, var, i) = getdetailsbincont!(M_t, df, var, i, ImputedContinuous) 

function getdetailsnoimpute!(M_t, df, var, i) 
    vec = getproperty(df, var)
    missings = Int[]
    return getdetails!(M_t, NoneImputed, df, var, vec, missings, i)
end 

function getdetailsbincont!(M_t, df, var, i, variabletype) 
    vec = getproperty(df, var)
    missings = identifymissings(vec)
    return getdetails!(M_t, variabletype, df, var, vec, missings, i)
end 

function getdetails!(M_t, variabletype, df, var, vec::Vector{<:Union{T, Missing}}, 
        missings, i
    ) where T <: AbstractString
    nmvec::Vector{T} = vec[Not(missings)]
    uniquevalues = unique(nmvec)
    @assert length(uniquevalues) == 2 """
        getdetails! currently only supports binary values. Variable $var has $(length(uniquevalues)) unique values
    """
    maxvalue = maximum(uniquevalues)
    minvalue = minimum(uniquevalues)
    floatnmvec = [ v == maxvalue ? 1. : .0 for v ∈ nmvec ]
    return getdetails!(M_t, vec, nmvec, var, i, variabletype, T, missings, floatnmvec, maxvalue, minvalue, 0, 0)
end 

function getdetails!(M_t, variabletype, df, var, vec::Vector{<:Union{T, Missing}}, missings, i) where T <: Number
    nmvec::Vector{T} = vec[Not(missings)]
    maxvalue = maximum(nmvec)
    minvalue = minimum(nmvec)
    return getdetails!(M_t, vec, nmvec, var, i, variabletype, T, missings, nmvec, "", "", maxvalue, minvalue)
end 

function getdetails!(M::Matrix, vec, nmvec, var, i, variabletype, T, missings, floatnmvec, 
        maxvaluestring, minvaluestring, maxvaluenumber, minvaluenumber
    )
    M[:, i] = setinitialvalues(vec, missings, nmvec, maxvaluenumber, floatnmvec)
    return getdetails(var, i, variabletype, T, missings, floatnmvec, 
        maxvaluestring, minvaluestring, maxvaluenumber, minvaluenumber)
end 

function getdetails!(tdf::DataFrame, vec, nmvec, var, i, variabletype, T, missings, floatnmvec, 
        maxvaluestring, minvaluestring, maxvaluenumber, minvaluenumber
    )
    insertcols!(tdf, var => setinitialvalues(vec, missings, nmvec, maxvaluenumber, floatnmvec))
    return getdetails(var, i, variabletype, T, missings, floatnmvec, 
        maxvaluestring, minvaluestring, maxvaluenumber, minvaluenumber)
end 

function getdetails(var, i, variabletype, T, missings, floatnmvec, 
        maxvaluestring, minvaluestring, maxvaluenumber, minvaluenumber
    )
    return VariableProperties(var, i, variabletype, T, missings, floatnmvec, 
        maxvaluestring, minvaluestring, maxvaluenumber, minvaluenumber)
end 

function setinitialvalues(vec, missings, nmvec::Vector{T}, maxvalue, floatnmvec) where T <: AbstractString
    currentvalues = zeros(length(vec))
    for i ∈ eachindex(vec)
        if i ∈ missings currentvalues[i] = Float64(sample(floatnmvec)) 
        else            currentvalues[i] = vec[i] == maxvalue ? 1. : .0
        end
    end 
    return currentvalues 
end 

function setinitialvalues(vec, missings, nmvec::Vector{T}, maxvalue, floatnmvec) where T <: Number
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Impute values 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# vec is either a vector of values used with the matrix M in the regression, or 
# a vector of FormulaTerm used in the regression with a DataFrame 

function impute(M::Matrix, vec::Vector{<:Number}, variableproperties, df; kwargs...) 
    # make copies so that imputations in different threads do not interfere with each other
    copyM = deepcopy(M) 
    copyvec = deepcopy(vec)
    return impute!(copyM, copyvec, variableproperties, df; kwargs...) 
end 

function impute(tdf::DataFrame, vec::Vector{<:FormulaTerm}, variableproperties, df; kwargs...) 
    # make copies so that imputations in different threads do not interfere with each other
    copytdf = deepcopy(tdf) 
    return impute!(copytdf, vec, variableproperties, df; kwargs...) 
end 

function impute!(M_t, vec, variableproperties, df; 
        initialvaluesfunc = sample, verbose, verbosei, kwargs...
    )
    if verbose @info "Starting imputation set $verbosei" end 
    initialvalues!(M_t, variableproperties, initialvaluesfunc)
    imputevalues!(M_t, vec, variableproperties; kwargs...)
    imputeddf = makeoutputdf(df, variableproperties, M_t)
    return imputeddf
end 

function initialvalues!(M_t, variableproperties::NamedTuple, initialvaluesfunc)
    # binary values are chosen using StatsBase.sample regardless of initialvaluesfunc
    initialvalues!(M_t, variableproperties.binarydict, sample)
    initialvalues!(M_t, variableproperties.contdict, initialvaluesfunc)
end 

function initialvalues!(M_t, dict::Dict, initialvaluesfunc)
    for k ∈ keys(dict)
        initialvalues!(M_t, dict[k], initialvaluesfunc)
    end 
end 

function initialvalues!(M::Matrix, variable::VariableProperties, initialvaluesfunc)
    i = variable.id
    initialvalues!(M, variable, initialvaluesfunc, i)
end 

function initialvalues!(tdf::DataFrame, variable::VariableProperties, initialvaluesfunc)
    name = variable.variablename
    initialvalues!(tdf, variable, initialvaluesfunc, name)
end 

function initialvalues!(M::Matrix, variable, initialvaluesfunc, i)
    for j ∈ variable.originalmissings
        M[j, i] = initialvaluesfunc(variable.nmvec)
    end 
end 

function initialvalues!(tdf::DataFrame, variable, initialvaluesfunc, name)
    for j ∈ variable.originalmissings
        getproperty(tdf, name)[j] = initialvaluesfunc(variable.nmvec)
    end 
end 

function imputevalues!(M_t, vec, variableproperties; m = 100)
    for _ ∈ 1:m _imputevalues!(M_t, vec, variableproperties) end 
end 

function _imputevalues!(M_t, vec, variableproperties)
    imputevaluesbin!(M_t, vec, getproperty(variableproperties, :binarydict))
    imputevaluescont!(M_t, vec, getproperty(variableproperties, :contdict))
end

function imputevaluesbin!(M_t, vec, dict::Dict)
    for k ∈ keys(dict) imputevaluesbin!(M_t, vec, dict[k]) end 
end 

function imputevaluesbin!(M::Matrix, vec, variable::VariableProperties)
    i = variable.id
    vec .= M[:, i]
    M[:, i] = ones(size(M, 1))
    regr = fit(GeneralizedLinearModel, M, vec, Binomial())
    probabilities = predict(regr)
    M[:, i] = vec
    imputevaluesbin!(M, variable, probabilities, i)
end 

function imputevaluesbin!(tdf::DataFrame, vec, variable::VariableProperties)
    i = variable.id
    fla = vec[i]
    regr = fit(GeneralizedLinearModel, fla, tdf, Binomial())
    probabilities = predict(regr)
    name = variable.variablename
    imputevaluesbin!(M, variable, probabilities, name)
end 

function imputevaluesbin!(M::Matrix, variable, probabilities, i)
    for j ∈ variable.originalmissings
        M[j, i] = rand() < probabilities[j] 
    end
end 

function imputevaluesbin!(tdf::DataFrame, variable, probabilities, name)
    for j ∈ variable.originalmissings
        getproperty(tdf, name)[j] = rand() < probabilities[j] 
    end
end 

function imputevaluescont!(M_t, vec, dict::Dict)
    for k ∈ keys(dict) imputevaluescont!(M_t, vec, dict[k]) end 
end 

function imputevaluescont!(M::Matrix, vec, variable::VariableProperties)
    i = variable.id
    vec .= M[:, i]
    M[:, i] = ones(size(M, 1))
    regr = fit(LinearModel, M, vec)
    predictions = predict(regr)
    M[:, i] = vec
    imputevaluescont!(M, variable, predictions, i)
end 

function imputevaluescont!(tdf::DataFrame, vec, variable::VariableProperties)
    i = variable.id
    fla = vec[i]
    regr = fit(LinearModel, fla, tdf)
    predictions = predict(regr)
    name = variable.variablename
    imputevaluescont!(tdf, variable, predictions, name)
end 

function imputevaluescont!(M::Matrix, variable, predictions, i)
    for j ∈ variable.originalmissings
        M[j, i] = predictions[j] 
    end
end 

function imputevaluescont!(tdf::DataFrame, variable, predictions, name)
    for j ∈ variable.originalmissings
        getproperty(tdf, name)[j] = predictions[j] 
    end
end 

function makeoutputdf(df, variableproperties, M_t)
    newdf = deepcopy(df)
    makeoutputdf!(newdf, variableproperties, M_t)
    return newdf
end 

function makeoutputdf!(newdf, variableproperties::NamedTuple, M_t)
    for d ∈ [ :binarydict, :contdict ]
        makeoutputdf!(newdf, getproperty(variableproperties, d), M_t)
    end
end 

function makeoutputdf!(newdf, dict::Dict, M_t)
    for k ∈ keys(dict) makeoutputdf!(newdf, dict[k], M_t) end 
end

function makeoutputdf!(newdf, variable::VariableProperties, M_t)
    select!(newdf, Not(variable.variablename))
    insertcols!(newdf, variable.variablename => imputedfvector(variable, M_t))
end 

function imputedfvector(variable, M_t)
    datatype = variable.datatype
    return imputedfvector(datatype, variable, M_t)
end 

function imputedfvector(datatype, variable, M_t) 
    if datatype <: AbstractString return imputestringvector(datatype, variable, M_t) 
    else                          return imputenumbervector(datatype, variable, M_t) 
    end
end 

function imputenumbervector(datatype, variable, M::Matrix)
    i = variable.id
    newvector::Vector{datatype} = M[:, i]
    return newvector
end 

function imputenumbervector(datatype, variable, tdf::DataFrame)
    name = variable.variablename
    newvector::Vector{datatype} = getproperty(tdf, name)
    return newvector
end 

function imputestringvector(datatype, variable, M_t)
    truestring = variable.truestring
    falsestring = variable.falsestring
    return imputestringvector(datatype, variable, M_t, truestring, falsestring)
end 

function imputestringvector(datatype, variable, M::Matrix, truestring, falsestring)
    i = variable.id
    newvector::Vector{datatype} = [ v == 1 ? truestring : falsestring for v ∈ M[:, i] ]
    return newvector
end 

function imputestringvector(datatype, variable, tdf::DataFrame, truestring, falsestring)
    name = variable.variablename
    newvector::Vector{datatype} = [ v == 1 ? truestring : falsestring for v ∈ getproperty(tdf, name) ]
    return newvector
end 
