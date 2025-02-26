
using SimpleMice
using Test
using DataFrames, GLM
using StableRNGs
import SimpleMice: MiceValue

@testset "SimpleMice.jl" begin
@testset "Equality and differences in MiceValues" begin
    @test MiceValue(2, [ 0, 0 ]) == MiceValue(2, [ 0, 0 ])
    @test MiceValue(2, [ 0, 0 ]) == MiceValue(2, [ 0.0, 0.0 ])
    @test MiceValue(2, [ 0, 0 ]) != MiceValue(2, [ 0, 1 ])
    @test MiceValue(2, [ 0, 0 ]) != MiceValue(3, [ 0, 0, 0 ])
    @test MiceValue(2, [ 1, 1 ]) ≈ MiceValue(2, [ 1.0, 1.0 ])
    @test !isapprox(MiceValue(2, [ 1, 1 ]), MiceValue(2, [ 1.0, 2.0 ]))
    @test MiceValue(2, [ 1, 1 ]) ≈ MiceValue(2, [ 1.0, 1 + 1e-20 ])
    @test isapprox(MiceValue(2, [ 0.1, 0.1 ]), MiceValue(2, [ 0.1, 0.15 ]); atol=0.05)
    @test isapprox(MiceValue(2, [ 0.1, 0.1 ]), MiceValue(2, [ 0.1, 0.15 ]); rtol=0.34)
end
@testset "Convert missing values into `Mice` values" begin
    # recognise vectors with no missing values 
    @test nonemissing(ones(2))
    # recognise vectors with at least one missing value 
    @test !nonemissing([ 1, missing ])
    # does not modify a vector with nothing missing
    @test initializemice(5, ones(1)) == ones(1)    
    # if nothing missing, returns same type as is provided 
    @test initializemice(5, ones(Int, 1)) isa Vector{Int64} 
    # values of missing data populated with other values from the vector 
    a1 = initializemice(5, [ 1, missing ])
    @test a1[2] == ones(5)
    # provides correct number of imputed values 
    a2 = initializemice(4, [ 1, missing ])
    @test a2[2] == ones(4)
    # imputed values of correct type 
    a3 = initializemice(5, [ Float64(1), missing ])
    @test a3[2] == ones(Float64, 5)
    # non-missing values are not changed 
    @test a1[1] == 1
    # converted in whichever order 
    a4 = initializemice(5, [ missing, 1 ])
    @test a4[1] == ones(Float64, 5)
    @test a4[2] == 1
    @testset "Test sampled values" begin
        rng = StableRNG(1)
        a = initializemice(rng, 10, [ 0, 1, 2, missing ])
        @test minimum(a[4]) == 0
        @test maximum(a[4]) == 2
    end
    @testset "same rng gives same results" begin
        rng_a = StableRNG(1)
        a = initializemice(rng_a, 10, [ 0, 1, 2, missing ])
        rng_b = StableRNG(1)
        b = initializemice(rng_b, 10, [ 0, 1, 2, missing ])        
        @test a == b
    end
    @testset "different rng gives different results" begin
        rng_a = StableRNG(1)
        a = initializemice(rng_a, 10, [ 0, 1, 2, missing ])
        rng_b = StableRNG(2)
        b = initializemice(rng_b, 10, [ 0, 1, 2, missing ])        
        @test a != b
    end
    # assertion error if no non-missing values 
    @test_throws AssertionError initializemice(1, [ missing ])
    # dimension mismatch error if wrong number of values passed 
    @test_throws DimensionMismatch MiceValue(5, [ 1, 2, 1 ]) 
end  
@testset "Convert missing values in a DataFrame into `Mice` values" begin
    @testset "DataFrame with no missing values" begin
        df = DataFrame(; a=[ 1, 2 ], b=[ 3.0, 4.0 ])
        dfa = initializemice(5, df)
        @testset for i ∈ 1:2, j ∈ 1:2
            @test df[i, j] == dfa[i, j]
        end
    end
    @testset "DataFrame with missing values in first column, column names" begin
        df = DataFrame(; a=[ 1, missing, 1 ], b=[ 3.0, 4.0, 5.0 ])
        dfa1 = initializemice(5, df)
        # convert missing values in first column
        @test dfa1.a[2] == [ 1.0, 1.0, 1.0, 1.0, 1.0 ]
        # no change to non-missing values 
        @testset for i ∈ 1:2, j ∈ 1:2
            if !ismissing(df[i, j])
                @test df[i, j] == dfa1[i, j]
            end
        end
        # choose not to convert column a so no changes to values 
        dfa2 = initializemice(5, df, "b")
        @testset for i ∈ 1:2, j ∈ 1:2
            @test ismissing(df[i, j]) == ismissing(dfa2[i, j])
            if !ismissing(df[i, j])
                @test df[i, j] == dfa2[i, j]
            end
        end
        # modify first column when given its name
        dfa3 = initializemice(5, df, "a")
        # convert missing values in first column
        @test dfa3.a[2] == [ 1.0, 1.0, 1.0, 1.0, 1.0 ]
        # no change to non-missing values 
        @testset for i ∈ 1:2, j ∈ 1:2
            if !ismissing(df[i, j])
                @test df[i, j] == dfa3[i, j]
            end
        end
        # modify first column when given both column names
        dfa4 = initializemice(5, df, [ "a", "b" ])
        @testset for i ∈ 1:2, j ∈ 1:2
            @test ismissing(dfa1[i, j]) == ismissing(dfa4[i, j])
            if !ismissing(df[i, j])
                @test dfa1[i, j] == dfa4[i, j]
            end
        end
    end
    @testset "DataFrame with missing values in first column, column symbols" begin
        df = DataFrame(; a=[ 1, missing, 1 ], b=[ 3.0, 4.0, 5.0 ])
        dfa1 = initializemice(5, df)
        # choose not to convert column a so no changes to values 
        dfa2 = initializemice(5, df, :b)
        @testset for i ∈ 1:2, j ∈ 1:2
            @test ismissing(df[i, j]) == ismissing(dfa2[i, j])
            if !ismissing(df[i, j])
                @test df[i, j] == dfa2[i, j]
            end
        end
        # modify first column when given its name
        dfa3 = initializemice(5, df, :a)
        # convert missing values in first column
        @test dfa3.a[2] == [ 1.0, 1.0, 1.0, 1.0, 1.0 ]
        # no change to non-missing values 
        @testset for i ∈ 1:2, j ∈ 1:2
            if !ismissing(df[i, j])
                @test df[i, j] == dfa3[i, j]
            end
        end
        # modify first column when given both column names
        dfa4 = initializemice(5, df, [ :a, :b ])
        @testset for i ∈ 1:2, j ∈ 1:2
            @test ismissing(dfa1[i, j]) == ismissing(dfa4[i, j])
            if !ismissing(df[i, j])
                @test dfa1[i, j] == dfa4[i, j]
            end
        end
    end
    @testset "DataFrame with missing values in first column, column indexes" begin
        df = DataFrame(; a=[ 1, missing, 1 ], b=[ 3.0, 4.0, 5.0 ])
        dfa1 = initializemice(5, df)
        # choose not to convert first column so no changes to values 
        dfa2 = initializemice(5, df, 2)
        @testset for i ∈ 1:2, j ∈ 1:2
            @test ismissing(df[i, j]) == ismissing(dfa2[i, j])
            if !ismissing(df[i, j])
                @test df[i, j] == dfa2[i, j]
            end
        end
        # modify first column when given its index
        dfa3 = initializemice(5, df, 1)
        # convert missing values in first column
        @test dfa3.a[2] == [ 1.0, 1.0, 1.0, 1.0, 1.0 ]
        # no change to non-missing values 
        @testset for i ∈ 1:2, j ∈ 1:2
            if !ismissing(df[i, j])
                @test df[i, j] == dfa3[i, j]
            end
        end
        # modify first column when given both column indexes
        dfa4 = initializemice(5, df, 1:2)
        @testset for i ∈ 1:2, j ∈ 1:2
            @test ismissing(dfa1[i, j]) == ismissing(dfa4[i, j])
            if !ismissing(df[i, j])
                @test dfa1[i, j] == dfa4[i, j]
            end
        end
    end
    @testset "DataFrame with no missing values and rng" begin
        df = DataFrame(; a=[ 1, 2 ], b=[ 3.0, 4.0 ])
        rng = StableRNG(1)
        dfa = initializemice(rng, 5, df)
        @testset for i ∈ 1:2, j ∈ 1:2
            @test df[i, j] == dfa[i, j]
        end
    end
    @testset "DataFrame with missing values in first column, with rng" begin
        df = DataFrame(; a=[ 1, missing, 1 ], b=[ 3.0, 4.0, 5.0 ])
        rng = StableRNG(1)
        dfa1 = initializemice(rng, 5, df)
        @test dfa1.a[2] == [ 1.0, 1.0, 1.0, 1.0, 1.0 ]
        # no change to non-missing values 
        @testset for i ∈ 1:2, j ∈ 1:2
            if !ismissing(df[i, j])
                @test df[i, j] == dfa1[i, j]
            end
        end
        # accept rng and String
        dfa2 = initializemice(rng, 5, df, "a")
        @test dfa2.a[2] == [ 1.0, 1.0, 1.0, 1.0, 1.0 ]
        # no change to non-missing values 
        @testset for i ∈ 1:2, j ∈ 1:2
            if !ismissing(df[i, j])
                @test df[i, j] == dfa2[i, j]
            end
        end
        # accept rng and vector of Strings
        dfa3 = initializemice(rng, 5, df, [ "a", "b" ])
        @testset for i ∈ 1:2, j ∈ 1:2
            @test ismissing(dfa1[i, j]) == ismissing(dfa3[i, j])
            if !ismissing(df[i, j])
                @test dfa1[i, j] == dfa3[i, j]
            end
        end
        # accept rng and Symbol
        dfa4 = initializemice(rng, 5, df, :a)
        @test dfa4.a[2] == [ 1.0, 1.0, 1.0, 1.0, 1.0 ]
        # no change to non-missing values 
        @testset for i ∈ 1:2, j ∈ 1:2
            if !ismissing(df[i, j])
                @test df[i, j] == dfa4[i, j]
            end
        end
        # accept rng and vector of Symbols
        dfa5 = initializemice(rng, 5, df, [ :a, :b ])
        @testset for i ∈ 1:2, j ∈ 1:2
            @test ismissing(dfa1[i, j]) == ismissing(dfa5[i, j])
            if !ismissing(df[i, j])
                @test dfa1[i, j] == dfa5[i, j]
            end
        end
        # accept rng and Index
        dfa6 = initializemice(rng, 5, df, 1)
        @test dfa6.a[2] == [ 1.0, 1.0, 1.0, 1.0, 1.0 ]
        # no change to non-missing values 
        @testset for i ∈ 1:2, j ∈ 1:2
            if !ismissing(df[i, j])
                @test df[i, j] == dfa6[i, j]
            end
        end
        # accept rng and vector of Indexes
        dfa7 = initializemice(rng, 5, df, 1:2)
        @testset for i ∈ 1:2, j ∈ 1:2
            @test ismissing(dfa1[i, j]) == ismissing(dfa7[i, j])
            if !ismissing(df[i, j])
                @test dfa1[i, j] == dfa7[i, j]
            end
        end
    end
    @testset "Same rng gives same results" begin
        @testset for i ∈ [ "a", [ "a", "b" ], :a, [ :a, :b ], 1, 1:2 ]
            df = DataFrame(; a=[ 1, missing, 2 ], b=[ 3.0, 4.0, 5.0 ])
            rng_a = StableRNG(1)
            dfa_a = initializemice(rng_a, 5, df, i)
            rng_b = StableRNG(1)
            dfa_b = initializemice(rng_b, 5, df, i)
            @test dfa_a.a == dfa_b.a  # confirm usefulness of test used in next section
            @testset for i ∈ 1:2, j ∈ 1:2
                @test ismissing(dfa_a[i, j]) == ismissing(dfa_b[i, j])
                if !ismissing(dfa_a[i, j])
                    @test dfa_a[i, j] == dfa_b[i, j]
                end
            end
        end
    end
    @testset "Different rng gives different results" begin
        @testset for i ∈ [ "a", [ "a", "b" ], :a, [ :a, :b ], 1, 1:2 ]
            df = DataFrame(; a=[ 1, missing, 2 ], b=[ 3.0, 4.0, 5.0 ])
            rng_a = StableRNG(1)
            dfa_a = initializemice(rng_a, 5, df, i)
            rng_b = StableRNG(2)
            dfa_b = initializemice(rng_b, 5, df, i)
            @test dfa_a.a != dfa_b.a
        end
    end
end  
@testset "Manipulate ImputedVector" begin
    rng_a = StableRNG(1)
    v = initializemice(rng_a, 5, [ 1, missing, 5 ])
    # imputed values [ 5.0, 1.0, 1.0, 5.0, 5.0 ]
    @test imputedvectorview(v, 1) == [ 1.0, 5.0, 5.0 ]
    @test imputedvectorview(v, 2) == [ 1.0, 1.0, 5.0 ]
    @test_throws AssertionError imputedvectorview(v, 0)
    @test_throws AssertionError imputedvectorview(v, 6)
    df = DataFrame( ; a=[ 1, missing, 2 ], b=[ 3, 4, 5 ])
    rng_b = StableRNG(1)
    dfa = initializemice(rng_b, 5, df)
    # imputed values [ 2.0, 1.0, 1.0, 2.0, 2.0 ]
    dfav1 = imputedtableview(dfa, 1)
    @test dfav1.a == [ 1.0, 2.0, 2.0 ]
    @test dfav1.b == [ 3, 4, 5 ]
    dfav2 = imputedtableview(dfa, 2)
    @test dfav2.a == [ 1.0, 1.0, 2.0 ]
    @test dfav2.b == [ 3, 4, 5 ]
    @test_throws AssertionError imputedtableview(dfa, 0)
    @test_throws AssertionError imputedtableview(dfa, 6)
end
@testset "Arithmetic on MiceValues" begin
    @test MiceValue(5, [ 0, 1, 2, 3, 4 ]) + 1 == MiceValue(5, [ 1, 2, 3, 4, 5 ])
    @test 1 + MiceValue(5, [ 0, 1, 2, 3, 4 ]) == MiceValue(5, [ 1, 2, 3, 4, 5 ])
    s = MiceValue(5, [ 1, 2, 1, 2, 1 ]) + MiceValue(5, [ 0, 1, 2, 3, 4 ])
    @test s == MiceValue(5, [ 1, 3, 3, 5, 5 ])
    @test_throws DimensionMismatch MiceValue(3, [ 1, 2, 1 ]) + MiceValue(4, [ 0, 1, 2, 3 ])
    @test MiceValue(5, [ 1, 2, 3, 4, 5 ]) - 1 == MiceValue(5, [ 0, 1, 2, 3, 4 ])
    @test 1 - MiceValue(5, [ 0, 1, 2, 3, 4 ]) == MiceValue(5, [ 1, 0, -1, -2, -3 ])
    s = MiceValue(5, [ 1, 3, 3, 5, 5 ]) - MiceValue(5, [ 0, 1, 2, 3, 4 ])
    @test MiceValue(5, [ 0, 1, 2, 3, 4 ]) * 2 == MiceValue(5, [ 0, 2, 4, 6, 8 ])
    @test MiceValue(5, [ 0, 1, 2, 3, 4 ]) / 2 == MiceValue(5, [ 0.0, 0.5, 1.0, 1.5, 2.0 ])
    @test MiceValue(5, [ 0, 1, 2, 3, 4 ]) ^ 2 == MiceValue(5, [ 0, 1, 4, 9, 16 ])
    @test s == MiceValue(5, [ 1, 2, 1, 2, 1 ])
    s2 = exp(MiceValue(5, [ 0, 1, 2, 3, 4 ]))
    @test s2 == MiceValue(5, [ exp(0), exp(1), exp(2), exp(3), exp(4) ])
    @test log(MiceValue(2, [ 1, 2 ])) == MiceValue(2, [ log(1), log(2) ])
    # next three tests use \approx as there were rounding errors. Also tests `isapprox`
    @test sin(MiceValue(2, [ π, 2 ])) ≈ MiceValue(2, [ sin(π), sin(2) ])
    @test cos(MiceValue(2, [ π, 2 ])) ≈ MiceValue(2, [ cos(π), cos(2) ])
    @test tan(MiceValue(2, [ π, 2 ])) ≈ MiceValue(2, [ tan(π), tan(2) ])
end
@testset "Update MiceValues, one imputed value, multiple predictive variables" begin
    # testing this function relies on assuming that GLM does what we want
    df = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [ 0.5, 5.0, missing, 4.0 ], 
        c = [ 2.0, missing, 10.9, 12.0 ]
    )
    rng_a = StableRNG(1)
    dfa = initializemice(rng_a, 5, df)
    # new values in column b: [ 4.0, 0.5, 5.0, 5.0, 0.5 ]
    # new values in column c: [ 2.0, 2.0, 12.0, 10.9, 2.0 ]
    temptable1 = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [ 0.5, 5.0, 4.0, 4.0 ], 
        c = [ 2.0, 2.0, 10.9, 12.0 ]
    )
    fla = @formula b ~ 1 + a + c
    regr1 = fit(LinearModel, fla, temptable1)
    predictions1 = predict(regr1)
    dfav1 = imputedtableview(dfa, 1)
    linearupdatemicevalues!(dfav1, :b, [ :a, :c ])
    # have the values in dfav1 changed as wanted? 
    @test dfav1.b[3] == predictions1[3]
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfav1[i, j]
    end 
    @test dfav1.c[2] == 2.0
    # have values in dfa changed as wanted?
    @test dfa.b[3] == [ predictions1[3], 0.5, 5.0, 5.0, 0.5 ] 
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfa[i, j]
    end 
    @test dfa.c[2] == [ 2.0, 2.0, 12.0, 10.9, 2.0 ]
    temptable2 = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [ 0.5, 5.0, 0.5, 4.0 ], 
        c = [ 2.0, 2.0, 10.9, 12.0 ]
    )
    regr2 = fit(LinearModel, fla, temptable2)
    predictions2 = predict(regr2)
    dfav2 = imputedtableview(dfa, 2)
    linearupdatemicevalues!(dfav2, :b, [ :a, :c ])
    # have the values in dfav1 changed as wanted? 
    @test dfav2.b[3] == predictions2[3]
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfav2[i, j]
    end 
    @test dfav2.c[2] == 2.0
    # have values in dfa changed as wanted?
    @test dfa.b[3] == [ predictions1[3], predictions2[3], 5.0, 5.0, 0.5 ] 
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfa[i, j]
    end 
    @test dfa.c[2] == [ 2.0, 2.0, 12.0, 10.9, 2.0 ]
    temptable3 = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [ 0.5, 5.0, 5.0, 4.0 ], 
        c = [ 2.0, 12.0, 10.9, 12.0 ]
    )
    regr3 = fit(LinearModel, fla, temptable3)
    predictions3 = predict(regr3)
    dfav3 = imputedtableview(dfa, 3)
    linearupdatemicevalues!(dfav3, "b", [ "a", "c" ])
    # have the values in dfav1 changed as wanted? 
    @test dfav3.b[3] == predictions3[3]
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfav3[i, j]
    end 
    @test dfav3.c[2] == 12.0
    # have values in dfa changed as wanted?
    @test dfa.b[3] == [ predictions1[3], predictions2[3], predictions3[3], 5.0, 0.5 ] 
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfa[i, j]
    end 
    @test dfa.c[2] == [ 2.0, 2.0, 12.0, 10.9, 2.0 ]
    temptable4 = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [ 0.5, 5.0, 5.0, 4.0 ], 
        c = [ 2.0, 10.9, 10.9, 12.0 ]
    )
    regr4 = fit(LinearModel, fla, temptable4)
    predictions4 = predict(regr4)
    dfav4 = imputedtableview(dfa, 4)
    linearupdatemicevalues!(dfav4, 2, [ 1, 3 ])
    # have the values in dfav1 changed as wanted? 
    @test dfav4.b[3] == predictions4[3]
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfav4[i, j]
    end 
    @test dfav4.c[2] == 10.9
    # have values in dfa changed as wanted?
    @test dfa.b[3] == [ 
        predictions1[3], 
        predictions2[3], 
        predictions3[3], 
        predictions4[3], 
        0.5 
    ] 
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfa[i, j]
    end 
    @test dfa.c[2] == [ 2.0, 2.0, 12.0, 10.9, 2.0 ]
    # final version included to allow test of `impute`
    dfav5 = imputedtableview(dfa, 5)
    linearupdatemicevalues!(dfav5, 2, [ 1, 3 ])
    rng_a = StableRNG(1)
    imputeresult1 = impute(rng_a, 5, df, :b, [ :a, :c ], 1)
    @test imputeresult1 == dfa
    rng_a = StableRNG(1)
    imputeresult2 = impute(rng_a, 5, df, "b", [ "a", "c" ], 1)
    @test imputeresult2 == dfa
    rng_a = StableRNG(1)
    imputeresult3 = impute(rng_a, 5, df, 2, [ 1, 3 ], 1)
    @test imputeresult3 == dfa
end
@testset "Update MiceValues, one imputed value, one predictive variable" begin
    # testing this function relies on assuming that GLM does what we want
    df = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [ 0.5, 5.0, missing, 4.0 ], 
        c = [ 2.0, missing, 10.9, 12.0 ]
    )
    rng_a = StableRNG(1)
    dfa = initializemice(rng_a, 5, df)
    # new values in column b: [ 4.0, 0.5, 5.0, 5.0, 0.5 ]
    # new values in column c: [ 2.0, 2.0, 12.0, 10.9, 2.0 ]
    temptable1 = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [ 0.5, 5.0, 4.0, 4.0 ], 
        c = [ 2.0, 2.0, 10.9, 12.0 ]
    )
    fla = @formula b ~ 1 + a
    regr1 = fit(LinearModel, fla, temptable1)
    predictions1 = predict(regr1)
    dfav1 = imputedtableview(dfa, 1)
    linearupdatemicevalues!(dfav1, :b, :a)
    # have the values in dfav1 changed as wanted? 
    @test dfav1.b[3] == predictions1[3]
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfav1[i, j]
    end 
    @test dfav1.c[2] == 2.0
    # have values in dfa changed as wanted?
    @test dfa.b[3] == [ predictions1[3], 0.5, 5.0, 5.0, 0.5 ] 
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfa[i, j]
    end 
    @test dfa.c[2] == [ 2.0, 2.0, 12.0, 10.9, 2.0 ]
    temptable2 = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [ 0.5, 5.0, 0.5, 4.0 ], 
        c = [ 2.0, 2.0, 10.9, 12.0 ]
    )
    regr2 = fit(LinearModel, fla, temptable2)
    predictions2 = predict(regr2)
    dfav2 = imputedtableview(dfa, 2)
    linearupdatemicevalues!(dfav2, :b, :a)
    # have the values in dfav1 changed as wanted? 
    @test dfav2.b[3] == predictions2[3]
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfav2[i, j]
    end 
    @test dfav2.c[2] == 2.0
    # have values in dfa changed as wanted?
    @test dfa.b[3] == [ predictions1[3], predictions2[3], 5.0, 5.0, 0.5 ] 
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfa[i, j]
    end 
    @test dfa.c[2] == [ 2.0, 2.0, 12.0, 10.9, 2.0 ]
    temptable3 = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [ 0.5, 5.0, 5.0, 4.0 ], 
        c = [ 2.0, 12.0, 10.9, 12.0 ]
    )
    regr3 = fit(LinearModel, fla, temptable3)
    predictions3 = predict(regr3)
    dfav3 = imputedtableview(dfa, 3)
    linearupdatemicevalues!(dfav3, "b", "a")
    # have the values in dfav1 changed as wanted? 
    @test dfav3.b[3] == predictions3[3]
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfav3[i, j]
    end 
    @test dfav3.c[2] == 12.0
    # have values in dfa changed as wanted?
    @test dfa.b[3] == [ predictions1[3], predictions2[3], predictions3[3], 5.0, 0.5 ] 
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfa[i, j]
    end 
    @test dfa.c[2] == [ 2.0, 2.0, 12.0, 10.9, 2.0 ]
    temptable4 = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [ 0.5, 5.0, 5.0, 4.0 ], 
        c = [ 2.0, 10.9, 10.9, 12.0 ]
    )
    regr4 = fit(LinearModel, fla, temptable4)
    predictions4 = predict(regr4)
    dfav4 = imputedtableview(dfa, 4)
    linearupdatemicevalues!(dfav4, 2, 1)
    # have the values in dfav1 changed as wanted? 
    @test dfav4.b[3] == predictions4[3]
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfav4[i, j]
    end 
    @test dfav4.c[2] == 10.9
    # have values in dfa changed as wanted?
    @test dfa.b[3] == [ 
        predictions1[3], 
        predictions2[3], 
        predictions3[3], 
        predictions4[3], 
        0.5 
    ] 
    # and has everything else stayed the same?
    for i ∈ 1:4, j ∈ 1:3 
        ismissing(df[i, j]) && continue 
        @test df[i, j] == dfa[i, j]
    end 
    @test dfa.c[2] == [ 2.0, 2.0, 12.0, 10.9, 2.0 ]
    # final version included to allow test of `impute`
    dfav5 = imputedtableview(dfa, 5)
    linearupdatemicevalues!(dfav5, 2, 1)
    rng_a = StableRNG(1)
    imputeresult1 = impute(rng_a, 5, df, :b, :a, 1)
    # cannot test whole table as column :c still has missing values 
    @test imputeresult1.a == dfa.a
    @test imputeresult1.b == dfa.b
    rng_a = StableRNG(1)
    imputeresult2 = impute(rng_a, 5, df, "b", "a", 1)
    @test imputeresult2.a == dfa.a
    @test imputeresult2.b == dfa.b
    rng_a = StableRNG(1)
    imputeresult3 = impute(rng_a, 5, df, 2, 1, 1)
    @test imputeresult3.a == dfa.a
    @test imputeresult3.b == dfa.b
end
@testset "Update MiceValues, multiple imputed values" begin
    # testing this function relies on assuming that GLM does what we want
    df = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [ 0.5, 5.0, missing, 4.0 ], 
        c = [ 2.0, missing, 10.9, 12.0 ]
    )
    rng_a = StableRNG(1)
    dfa = initializemice(rng_a, 5, df)
    # this test relies on `linearupdatemicevalues!` working properly -- it was tested above
    for i ∈ 1:5 
        dfav = imputedtableview(dfa, i)
        linearupdatemicevalues!(dfav, :b, [ :a, :c ])
        linearupdatemicevalues!(dfav, :c, [ :a, :b ])
    end
    rng_a = StableRNG(1)
    imputeresult1 = impute(rng_a, 5, df, [ :b, :c ], :a, 1)
    rng_a = StableRNG(1)
    imputeresult2 = impute(rng_a, 5, df, [ "b", "c" ], "a", 1)
    rng_a = StableRNG(1)
    imputeresult3 = impute(rng_a, 5, df, 2:3, 1, 1)
    # answers differ at the 15th decimal point 
    for i ∈ 1:4, j ∈ 1:3 
        @test isapprox(imputeresult1[i, j], dfa[i, j]; atol=1e-12)
        @test isapprox(imputeresult2[i, j], dfa[i, j]; atol=1e-12)
        @test isapprox(imputeresult3[i, j], dfa[i, j]; atol=1e-12)
    end
end
@testset "Multithread argument does not change output" begin
    df = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [ 0.5, 5.0, missing, 4.0 ], 
        c = [ 2.0, missing, 10.9, 12.0 ]
    )
    rng_a = StableRNG(1)
    imputeresult1 = impute(rng_a, 5, df, [ :b, :c ], :a, 1)
    rng_a = StableRNG(1)
    imputeresult2 = impute(rng_a, 5, df, [ :b, :c ], :a, 1; multithread=true)
    rng_a = StableRNG(1)
    imputeresult3 = impute(rng_a, 5, df, [ :b, :c ], :a, 1; multithread=false)
    for i ∈ 1:4, j ∈ 1:3 
        @test isapprox(imputeresult1[i, j], imputeresult2[i, j]; atol=1e-12)
        @test isapprox(imputeresult2[i, j], imputeresult3[i, j]; atol=1e-12)
    end
end
@testset "Effect of `staticthresh` keyword" begin
    df = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [ 0.5, 5.0, missing, 4.0 ], 
        c = [ 2.0, missing, 10.9, 12.0 ]
    )
    rng_a = StableRNG(1)
    imputeresult1 = impute(rng_a, 5, df, [ :b, :c ], :a, 1)
    rng_a = StableRNG(1)
    imputeresult2 = impute(rng_a, 5, df, [ :b, :c ], :a, 1; staticthresh=0)
    rng_a = StableRNG(1)
    imputeresult3 = impute(rng_a, 5, df, [ :b, :c ], :a, 1; staticthresh=3)
    rng_a = StableRNG(1)
    imputeresult4 = impute(rng_a, 5, df, [ :b, :c ], :a, 1; staticthresh=500)
    # staticthresh keyword does not affect outcome
    for i ∈ 1:4, j ∈ 1:3 
        @test isapprox(imputeresult1[i, j], imputeresult2[i, j]; atol=1e-12)
        @test isapprox(imputeresult2[i, j], imputeresult3[i, j]; atol=1e-12)
        @test isapprox(imputeresult3[i, j], imputeresult4[i, j]; atol=1e-12)
    end
    # staticthresh keyword does affect type of output 
    @test imputeresult1.b isa SimpleMice.ImputedVectorMStatic
    @test imputeresult2.b isa SimpleMice.ImputedVector
    @test imputeresult3.b isa SimpleMice.ImputedVector
    @test imputeresult4.b isa SimpleMice.ImputedVectorMStatic
end
end  # @testset "SimpleMice.jl"
