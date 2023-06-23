
"""
    fit(model::Type{T}, formula::FormulaTerm, idf::ImputedDataFrame, 
        <additional arguments>; <keyword arguments>) where T <: RegressionModel

Apply the model to each imputed dataset, then produce summary statistics across all
    imputed results. 

Results currently have p-values that are too small so should not be used.
"""
function fit(model::Type{T}, formula::FormulaTerm, idf::ImputedDataFrame, args...; 
        kwargs...
    ) where T <: RegressionModel
    n = idf.numberimputed
    samplesize = size(idf.imputeddfs[1], 1)
    regrs = [ fit(model, formula, idf.imputeddfs[i], args...; kwargs...) for i ∈ 1:n ]
    coefnamelist = coefnames(regrs[1])
    coefficientmatrix = makreregressionmatrix(coef, regrs, coefnamelist, n)
    coefs = rubinscoefficients(coefficientmatrix, n)
    varmatrix = makevarmatrix(regrs, coefs, n, samplesize)
    vars = rubinsvars(coefficientmatrix, varmatrix, n)
    sterrors = vartosterror.(vars, n)
    tcalcs = [ OneSampleTTest(coefs[i], sterrors[i], samplesize) for i ∈ eachindex(coefs) ]
    tvalues = [ tcalcs[i].t for i ∈ eachindex(coefs) ]
    pvalues = pvalue.(tcalcs)
    confints = confint.(tcalcs)
    resultsmatrix = makeresultsmatrix(coefnamelist, coefs, sterrors, tvalues, pvalues, confints)
    printfit(resultsmatrix, formula, n)
    return ( coefnames = coefnamelist, coef = coefs, stderror = sterrors, t = tvalues, 
        pvalue = pvalues, confint = confints )
end 

"""
    lm(formula::FormulaTerm, idf::ImputedDataFrame, <additional arguments>; <keyword arguments>)

Fit a generalized linear model to data. Alias for 
    `fit(LinearModel, formula, idf, <additional arguments>; <keyword arguments>)`.
"""
function lm(formula::FormulaTerm, idf::ImputedDataFrame, args...; kwargs...) 
    return fit(LinearModel, formula, idf, args...; kwargs...) 
end 

"""
    glm(formula::FormulaTerm, idf::ImputedDataFrame, distr::UnivariateDistribution, 
        link::Link = canonicallink(distr); <keyword arguments>)

Fit a generalized linear model to data. Alias for 
    `fit(GeneralizedLinearModel, formula, idf, distr, link; <keyword arguments>)`.
"""
function glm(formula::FormulaTerm, idf::ImputedDataFrame, distr::UnivariateDistribution, 
        link::Link = canonicallink(distr); kwargs...
    )
    return fit(GeneralizedLinearModel, formula, idf, distr, link; kwargs...)
end 

function rubinscoefficients(coefficientmatrix, n)
    coefs = [ rubinsmean(coefficientmatrix[i, :], n) for i ∈ axes(coefficientmatrix, 1) ]
    return coefs 
end 

function rubinsvars(coefficientmatrix, varmatrix, n)
    vars = [ rubinsvar(coefficientmatrix[i, :], varmatrix[i, :], n) 
        for i ∈ axes(coefficientmatrix, 1) ]
    return vars
end 

function makreregressionmatrix(func, regrs, coefnamelist, n)
    regrmatrix = zeros(length(coefnamelist), n)
    for i ∈ 1:n regrmatrix[:, i] = func(regrs[i]) end
    return regrmatrix
end 

function makevarmatrix(regrs, coefs, n, samplesize)
    stderrormatrix = makreregressionmatrix(stderror, regrs, coefs, n)
    varmatrix = sterrortovar.(stderrormatrix, samplesize) 
    return varmatrix 
end 

function makeresultsmatrix(coefnamelist, coefs, sterrors, tvalues, pvalues, confints)
    results = Array{Any, 2}(undef, length(coefnamelist), 7)
    for i ∈ eachindex(coefnamelist)
        for (j, t) ∈ enumerate([ coefnamelist, coefs, sterrors, tvalues, pvalues ])
            results[i, j] = t[i] 
        end 
        results[i, 6] = confints[i][1]
        results[i, 7] = confints[i][2]
    end 
    return results
end 

function printfit(resultsmatrix, formula, n)
    println("Regression results from $n imputed datasets")
    println("$formula")
    @pt :header = regressionheadings resultsmatrix
end 
