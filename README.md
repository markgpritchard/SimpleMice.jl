# SimpleMice.jl

`SimpleMice.jl` takes a `DataFrame` with missing values and produces a collection of datasets with imputed values.

Missing values can be binary or continuous. Initially, all missing values are replaced by sampling from the non-missing values. A series of regressions are then performed for each variable with missing data. For continuous variables, linear regression is performed and predicted values from the regression are used for the missing values. For binary variables, logistic regression is performed and predicted values are used to generate a probability. The imputed value is then determined by `rand() < probability`.

## Installation
```julia
julia> using Pkg

julia> Pkg.add(url = "https://github.com/markgpritchard/SimpleMice.jl")
```

## Example 
Make some sample data with values missing completely at random.
```julia
julia> using SimpleMice

julia> testdata = SimpleMice.testdataset()
1000×9 DataFrame
  Row │ Ages      Sexes    Vara   Varb    Varc   Vard      Vare     Varf   Outcome 
      │ Float64?  String?  Bool?  Int64?  Bool?  Float64?  String?  Bool?  Bool    
──────┼────────────────────────────────────────────────────────────────────────────
    1 │     2.66  F        false       0  false   29.5344  N        false    false
    2 │     1.55  M        false       0  false   29.5971  Y        false    false
    3 │     1.77  M        false       0  false   28.1784  N        false    false
  ⋮   │    ⋮         ⋮       ⋮      ⋮       ⋮       ⋮         ⋮       ⋮       ⋮
  999 │    93.39  M        false       0  false   22.9344  Y         true    false
 1000 │    92.75  F        false       0  false   27.4594  Y         true    false
                                                                   995 rows omitted

julia> mcar!(testdata, [ :Vara, :Varb, :Varc, :Vard ], 0.2)

julia> micedata = mice(testdata)
SimpleMice.ImputedDataFrame(1000×9 DataFrame
  Row │ Ages      Sexes    Vara     Varb     Varc     Vard          Vare     Varf   Outcome 
      │ Float64?  String?  Bool?    Int64?   Bool?    Float64?      String?  Bool?  Bool    
──────┼─────────────────────────────────────────────────────────────────────────────────────
    1 │     2.66  F        missing        0  missing       29.5344  N        false    false
    2 │     1.55  M        missing        0    false       29.5971  Y        false    false
    3 │     1.77  M          false  missing    false       28.1784  N        false    false
  ⋮   │    ⋮         ⋮        ⋮        ⋮        ⋮          ⋮           ⋮       ⋮       ⋮
  999 │    93.39  M          false        0  missing       22.9344  Y         true    false
 1000 │    92.75  F          false        0    false  missing       Y         true    false
                                                                            995 rows omitted, 5, DataFrames.DataFrame[1000×9 DataFrame
  Row │ Ages      Sexes    Vare     Varf   Outcome  Vara   Varb   Varc   Vard    
      │ Float64?  String?  String?  Bool?  Bool     Bool   Int64  Bool   Float64 
──────┼──────────────────────────────────────────────────────────────────────────
    1 │     2.66  F        N        false    false  false      0  false  29.5344
    2 │     1.55  M        Y        false    false  false      0  false  29.5971
    3 │     1.77  M        N        false    false  false      0  false  28.1784
  ⋮   │    ⋮         ⋮        ⋮       ⋮       ⋮       ⋮      ⋮      ⋮       ⋮
  999 │    93.39  M        Y         true    false  false      0  false  22.9344
 1000 │    92.75  F        Y         true    false  false      0  false  24.8777
                                                                 995 rows omitted, 1000×9 DataFrame
  Row │ Ages      Sexes    Vare     Varf   Outcome  Vara   Varb   Varc   Vard    
      │ Float64?  String?  String?  Bool?  Bool     Bool   Int64  Bool   Float64 
──────┼──────────────────────────────────────────────────────────────────────────
    1 │     2.66  F        N        false    false  false      0  false  29.5344
    2 │     1.55  M        Y        false    false  false      0  false  29.5971
    3 │     1.77  M        N        false    false  false      0  false  28.1784
  ⋮   │    ⋮         ⋮        ⋮       ⋮       ⋮       ⋮      ⋮      ⋮       ⋮
  999 │    93.39  M        Y         true    false  false      0  false  22.9344
 1000 │    92.75  F        Y         true    false  false      0  false  24.8326
                                                                 995 rows omitted, 1000×9 DataFrame
  Row │ Ages      Sexes    Vare     Varf   Outcome  Vara   Varb   Varc   Vard    
      │ Float64?  String?  String?  Bool?  Bool     Bool   Int64  Bool   Float64 
──────┼──────────────────────────────────────────────────────────────────────────
    1 │     2.66  F        N        false    false  false      0  false  29.5344
    2 │     1.55  M        Y        false    false  false      0  false  29.5971
    3 │     1.77  M        N        false    false  false      0  false  28.1784
  ⋮   │    ⋮         ⋮        ⋮       ⋮       ⋮       ⋮      ⋮      ⋮       ⋮
  999 │    93.39  M        Y         true    false  false      0   true  22.9344
 1000 │    92.75  F        Y         true    false  false      0  false  24.9394
                                                                 995 rows omitted, 1000×9 DataFrame
  Row │ Ages      Sexes    Vare     Varf   Outcome  Vara   Varb   Varc   Vard    
      │ Float64?  String?  String?  Bool?  Bool     Bool   Int64  Bool   Float64 
──────┼──────────────────────────────────────────────────────────────────────────
    1 │     2.66  F        N        false    false  false      0  false  29.5344
    2 │     1.55  M        Y        false    false  false      0  false  29.5971
    3 │     1.77  M        N        false    false  false      1  false  28.1784
  ⋮   │    ⋮         ⋮        ⋮       ⋮       ⋮       ⋮      ⋮      ⋮       ⋮
  999 │    93.39  M        Y         true    false  false      0  false  22.9344
 1000 │    92.75  F        Y         true    false  false      0  false  24.9713
                                                                 995 rows omitted, 1000×9 DataFrame
  Row │ Ages      Sexes    Vare     Varf   Outcome  Vara   Varb   Varc   Vard    
      │ Float64?  String?  String?  Bool?  Bool     Bool   Int64  Bool   Float64 
──────┼──────────────────────────────────────────────────────────────────────────
    1 │     2.66  F        N        false    false  false      0  false  29.5344
    2 │     1.55  M        Y        false    false  false      0  false  29.5971
    3 │     1.77  M        N        false    false  false      0  false  28.1784
  ⋮   │    ⋮         ⋮        ⋮       ⋮       ⋮       ⋮      ⋮      ⋮       ⋮
  999 │    93.39  M        Y         true    false  false      0  false  22.9344
 1000 │    92.75  F        Y         true    false  false      0  false  24.9762
                                                                 995 rows omitted])
```

The result comes in a structure `ImputedDataFrame` which holds the original `DataFrame`, the number of imputed datasets and a vector of `DataFrame`s with imputed values.

There are currently a small set of functions to explore the imputed datasets.
```julia
julia> mean(getvalues(micedata, :Vard))
24.631066907927504

julia> describe(micedata)
9×7 DataFrame
 Row │ variable  mean     min      median   max      nmissing  eltype
     │ Symbol    Union…   Any      Union…   Any      Int64     Type
─────┼─────────────────────────────────────────────────────────────────────────────────
   1 │ Ages      47.9968  0.5      48.005   94.9            0  Union{Missing, Float64} 
   2 │ Sexes              F                 M               0  Union{Missing, String}  
   3 │ Vara      0.1162   0.0      0.0      1.0             0  Bool
   4 │ Varb      0.2248   0.0      0.0      1.0             0  Int64
   5 │ Varc      0.0122   0.0      0.0      1.0             0  Bool
   6 │ Vard      24.6311  8.99611  24.7306  39.5763         0  Float64
   7 │ Vare               N                 Y               0  Union{Missing, String}  
   8 │ Varf      0.549    0.0      1.0      1.0             0  Union{Missing, Bool}    
   9 │ Outcome   0.061    0.0      0.0      1.0             0  Bool
```

We can also fit a regression model to the imputed dataset.
```julia
julia> using GLM

julia> formula = @formula Outcome ~ Vara + Varb + Varc + Vard

julia> lm(formula, micedata)
Regression results from 5 imputed datasets

Outcome ~ Vara + Varb + Varc + Vard

Coefficients:
5×7 DataFrame
 Row │              Coef         Std. Error  t          Pr(>|t|)    Lower 95%    Upper 95%   
     │ Any          Any          Any         Any        Any         Any          Any
─────┼───────────────────────────────────────────────────────────────────────────────────────
   1 │ (Intercept)  0.107146     0.0442529   2.42122    0.0156458   0.0203067    0.193985
   2 │ Vara         0.0905258    0.0274943   3.29253    0.00102771  0.0365726    0.144479
   3 │ Varb         0.0580355    0.0221131   2.62448    0.00881048  0.014642     0.101429
   4 │ Varc         -0.0466178   0.0689097   -0.676505  0.498876    -0.181842    0.0886065
   5 │ Vard         -0.00280791  0.00174399  -1.61005   0.107703    -0.00623022  0.000614397
```

Note that when applied to an `ImputedDataFrame`, `fit` and `glm` assume the parameters can be combined as a mean according to Rubin's rules. You should check that this is appropriate for the type of regression being performed.

## To do
Key issues include:
* Adding ability to impute missing data for categorical variables.
* Adding function to combine regression parameters when a simple mean from Rubin's rules is not appropriate.
* Supplying a more realistic model dataset.
* Efforts to improve speed for large datasets.
* Comparison of the output from this function to other packages, such as `mice.R`.
