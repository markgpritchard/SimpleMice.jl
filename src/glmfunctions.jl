
"""
    fit(model::Type{T}, formula::FormulaTerm, idf::ImputedDataFrame, 
        <additional arguments>; <keyword arguments>) where T <: RegressionModel

Apply the model to each imputed dataset, then produce summary statistics across all
    imputed results. 

Note that when applied to an `ImputedDataFrame` this function assumes the parameters 
    can be combined as a mean according to Rubin's rules. You should check that 
    this is appropriate for the type of regression being performed.
"""
function fit(model::Type{T}, formula::FormulaTerm, idf::ImputedDataFrame, args...; 
        kwargs...
    ) where T <: RegressionModel
    n = idf.numberimputed
    samplesize = size(idf.imputeddfs[1], 1)
    regrs = [ fit(model, formula, idf.imputeddfs[i], args...; kwargs...) for i ∈ 1:n ]
    return processfit(regrs, formula, n, samplesize)
end 

"""
    lm(formula::FormulaTerm, idf::ImputedDataFrame, <additional arguments>; <keyword arguments>)

Fit a linear model to data. Alias for 
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

Note that when applied to an `ImputedDataFrame` this function assumes the parameters 
    can be combined as a mean according to Rubin's rules. You should check that 
    this is appropriate for the type of regression being performed.
"""
function glm(formula::FormulaTerm, idf::ImputedDataFrame, distr::UnivariateDistribution, 
        link::Link = canonicallink(distr); kwargs...
    )
    return fit(GeneralizedLinearModel, formula, idf, distr, link; kwargs...)
end 

function processfit(regrs, formula, n, samplesize)
    coefnamelist = coefnames(regrs[1])
    coefs, coefficientmatrix = fitcoefficients(regrs, coefnamelist, n)
    stderrors = fitstderrors(regrs, coefnamelist, n, coefficientmatrix)
    tvalues, tcalcs = fittvalues(coefs, stderrors, samplesize)
    pvalues = pvalue.(tcalcs)
    confints = confint.(tcalcs)
    regressionresults = ImputedRegressionResult(
        formula, n, coefnamelist, coefs, stderrors, tvalues, pvalues, confints
    )
    printfit(regressionresults)
    return regressionresults 
end 

function fitcoefficients(regrs, coefnamelist, n)
    coefficientmatrix = makreregressionmatrix(coef, regrs, coefnamelist, n)
    coefs = rubinscoefficients(coefficientmatrix, n)
    return ( coefs, coefficientmatrix )
end 

function fitstderrors(regrs, coefnamelist, n, coefficientmatrix)
    stderrormatrix = makreregressionmatrix(stderror, regrs, coefnamelist, n)
    return rubinsstderrors(coefficientmatrix, stderrormatrix, n) 
end 

function fittvalues(coefs, sterrors, samplesize)
    tcalcs = [ OneSampleTTest(coefs[i], sterrors[i] * sqrt(samplesize), samplesize) 
        for i ∈ eachindex(coefs) ]
    tvalues = [ tcalcs[i].t for i ∈ eachindex(coefs) ]
    return ( tvalues, tcalcs )
end 

function rubinscoefficients(coefficientmatrix, n)
    coefs = [ rubinsmean(coefficientmatrix[i, :], n) for i ∈ axes(coefficientmatrix, 1) ]
    return coefs 
end 

function rubinsstderrors(coefficientmatrix, stderrormatrix, n)
    stderrors = [ rubinssterror(coefficientmatrix[i, :], stderrormatrix[i, :], n) 
        for i ∈ axes(coefficientmatrix, 1) ]
    return stderrors
end 

function makreregressionmatrix(func, regrs, coefnamelist, n)
    regrmatrix = zeros(length(coefnamelist), n)
    for i ∈ 1:n regrmatrix[:, i] = func(regrs[i]) end
    return regrmatrix
end 

function makeresultsmatrix(regressionresults)
    ℓ = length(regressionresults.coefnames)
    results = Array{Any, 2}(undef, ℓ, 7)
    for i ∈ 1:ℓ
        for (j, v) ∈ enumerate([ :coefnames, :coef, :stderror, :t, :pvalue ])
            results[i, j] = getproperty(regressionresults, v)[i]
        end 
        results[i, 6] = regressionresults.confint[i][1]
        results[i, 7] = regressionresults.confint[i][2]
    end 
    return results
end 

function makeresultsdataframe(regressionresults::ImputedRegressionResult)
    resultsmatrix = makeresultsmatrix(regressionresults)
    regressionheadings = [ 
        "", "Coef", "Std. Error", "t", "Pr(>|t|)", "Lower 95%", "Upper 95%" 
    ]
    resultsdf = DataFrame([ head => resultsmatrix[:, i] 
        for (i, head) ∈ enumerate(regressionheadings) ])
    return resultsdf
end 

function printfit(regressionresults::ImputedRegressionResult)
    resultsdf = makeresultsdataframe(regressionresults)
    printfit(regressionresults, resultsdf)
end 

function printfit(regressionresults::ImputedRegressionResult, resultsdf::DataFrame)
    println("Regression results from $(regressionresults.n) imputed datasets")
    println("")
    println("$(regressionresults.formula)")
    println("")
    println("Coefficients:")
    show(stdout, resultsdf)
end 
