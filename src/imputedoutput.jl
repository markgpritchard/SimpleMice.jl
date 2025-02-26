# `ImputedOutput` is used to hold outputs of analyses with imputed values

abstract type AbstractImputedOutput{Ni} end  # not exported

@auto_hash_equals struct ImputedOutput{Ni, S, T} <: AbstractImputedOutput{Ni}  # not exported
    imputedsums         :: SVector{Ni, S} 
    imputedmeans        :: SVector{Ni, T} 
    imputedvars         :: SVector{Ni, T}
    rubinsmean          :: T
    rubinsvar           :: T
end

function summaryimputedoutput(v::AbstractImputedVector)
    imputedsums = sum(v)
    imputedmeans = elementmean(v)
    imputedvars = elementvar(v)
    rubinsmean = mean(v)
    rubinv = rubinsvar(v)
    return ImputedOutput{Ni, typeof(imputedsums[1]), typeof(rubinsmean)}(
        imputedsums, imputedmeans, imputedvars, rubinsmean, rubinv
    )
end
