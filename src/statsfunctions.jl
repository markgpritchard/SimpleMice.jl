
# Source for these functions: 
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2727536/

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Combinations by Rubin's rules 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
    rubinsmean(q::Vector, n::Int)

Combines the mean from each imputed dateset and produces an overall mean according 
    to Rubin's rules.

`q` is a vector of the mean from each imputed dataset and `n` is the number of imputed 
    datasets.

Note, this function does not test whether `length(q) == n`.
"""
rubinsmean(q::Vector, n::Int) = sum(q) / n

"""
    function rubinsvar(q::Vector, u::Vector, n::Int)

Calculates a variance from multiple imputed datasets according to Rubin's rules. 

`q` is a vector of the mean from each imputed dataset, `u` is a vector of the variance 
    of each imputed dataset, and `n` is the number of imputed datasets.

Note, this function does not test whether `length(q) == n`.
"""
function rubinsvar(q::Vector, u::Vector, n::Int)
    ubar = withinimputationvar(u, n)
    b = betweenimputationvar(q, u, n)
    v = ubar + (1 + 1 / n) * b 
    return v 
end 

"""
    withinimputationvar(u::Vector, n::Int) = sum(u) / n 

Calculates a mean within-imputation variance from multiple imputed datasets. 

`u` is a vector of the variance of each imputed dataset and `n` is the number of 
    imputed datasets.

Note, this function does not test whether `length(q) == n`.
"""
withinimputationvar(u::Vector, n::Int) = sum(u) / n 

"""
    betweenimputationvar(q::Vector, u::Vector, n::Int)

Calculates between-imputation variance from multiple imputed datasets. 

`q` is a vector of the mean from each imputed dataset, `u` is a vector of the variance 
    of each imputed dataset, and `n` is the number of imputed datasets.

Note, this function does not test whether `length(q) == n`.
"""
function betweenimputationvar(q::Vector, u::Vector, n::Int)
    qbar = rubinsmean(q, n)
    qsquarediff = @. (q - qbar)^2
    b = sum(qsquarediff) / (n - 1)
    return b 
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions used in calculating combined statistics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
    componentmeans(d::ImputedVector)

Produces a `Vector` of the mean value of each imputed vector. 
"""
componentmeans(d::ImputedVector) = [ mean(d.imputedvalues[i]) for i ∈ 1:d.numberimputed ]

"""
    componentvars(d::ImputedVector)

Produces a `Vector` of the variance of each imputed vector. 
"""
componentvars(d::ImputedVector) = [ var(d.imputedvalues[i]) for i ∈ 1:d.numberimputed ]

function sterrortovar(sterror, n)
    std = sterror * sqrt(n) 
    var = std^2 
    return var 
end 

function vartosterror(var, n)
    std = sqrt(var) 
    sterror = std / sqrt(n)
    return sterror 
end 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Versions of imported functions applied to imputed datasets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
    mean(d::ImputedVector)

Calculate the mean from multiple imputed datasets and combine them according to Rubin's 
    rules.
"""
function mean(d::ImputedVector)
    cm = componentmeans(d)
    return rubinsmean(cm, d.numberimputed)
end 

"""
    var(d::ImputedVector)

Calculate the variance from multiple imputed datasets according to Rubin's rules.
"""
function var(d::ImputedVector)
    cm = componentmeans(d)
    cv = componentvars(d)
    return rubinsvar(cm, cv, d.numberimputed)
end 

"""
    std(d::ImputedVector)

Calculate the standard deviation from multiple imputed datasets as `sqrt(var(d))`. 
"""
std(d::ImputedVector) = sqrt(var(d))
