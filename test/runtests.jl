
using SimpleMice
using Test
using DataFrames
using StableRNGs

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
    @test initializemice(5, [ 1, missing ]) == [ 1, MiceValue(5, ones(Int, 5)) ]
    # provides correct number of imputed values 
    @test initializemice(4, [ 1, missing ]) == [ 1, MiceValue(4, ones(Int, 4)) ]
    # imputed values of correct type 
    @test initializemice(5, [ Float64(1), missing ]) == [ 1, MiceValue(5, ones(Float64, 5)) ]
    # non-missing values are not changed 
    @test initializemice(5, [ 2, missing ]) == [ 2, MiceValue(5, 2 .* ones(Int, 5)) ]
    # converted in whichever order 
    @test initializemice(5, [ missing, 1 ]) == [ MiceValue(5, ones(Int, 5)), 1 ]
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
        dft = deepcopy(df)
        initializemice!(5, dft)
        @test dft == df
    end
    @testset "DataFrame with missing values in first column, column names" begin
        df = DataFrame(; a=[ 1, missing, 1 ], b=[ 3.0, 4.0, 5.0 ])
        dft = deepcopy(df)
        initializemice!(5, dft)
        # convert missing values in first column
        @test dft == DataFrame(; a=[ 1, MiceValue(5, ones(Int, 5)), 1 ], b=[ 3.0, 4.0, 5.0 ])
        # do nothing when given name of column with no missing values 
        dft2 = deepcopy(df)
        initializemice!(5, dft2, "b")
        # need to subdivide test as missing == missing -> missing 
        @test dft2.b == df.b
        @test dft2.a[1] == df.a[1]
        @test ismissing(dft2.a[2])
        @test dft2.a[3] == df.a[3]
        # modify first column when given its name
        dft3 = deepcopy(df)
        initializemice!(5, dft3, "a")
        @test dft3 == dft
        # modify first column when given both column names
        dft4 = deepcopy(df)
        initializemice!(5, dft4, [ "a", "b" ])
        @test dft4 == dft
    end
    @testset "DataFrame with missing values in first column, column symbols" begin
        df = DataFrame(; a=[ 1, missing, 1 ], b=[ 3.0, 4.0, 5.0 ])
        dft = deepcopy(df)
        initializemice!(5, dft)
        # do nothing when given name of column with no missing values 
        dft2 = deepcopy(df)
        initializemice!(5, dft2, :b)
        # need to subdivide test as missing == missing -> missing 
        @test dft2.b == df.b
        @test dft2.a[1] == df.a[1]
        @test ismissing(dft2.a[2])
        @test dft2.a[3] == df.a[3]
        # modify first column when given its name
        dft3 = deepcopy(df)
        initializemice!(5, dft3, :a)
        @test dft3 == dft
        # modify first column when given both column names
        dft4 = deepcopy(df)
        initializemice!(5, dft4, [ :a, :b ])
        @test dft4 == dft
    end
    @testset "DataFrame with missing values in first column, column indexes" begin
        df = DataFrame(; a=[ 1, missing, 1 ], b=[ 3.0, 4.0, 5.0 ])
        dft = deepcopy(df)
        initializemice!(5, dft)
        # do nothing when given index of column with no missing values 
        dft2 = deepcopy(df)
        initializemice!(5, dft2, 2)
        # need to subdivide test as missing == missing -> missing 
        @test dft2.b == df.b
        @test dft2.a[1] == df.a[1]
        @test ismissing(dft2.a[2])
        @test dft2.a[3] == df.a[3]
        # modify first column when given its index
        dft3 = deepcopy(df)
        initializemice!(5, dft3, 1)
        @test dft3 == dft
        # modify first column when given both column indexes
        dft4 = deepcopy(df)
        initializemice!(5, dft4, 1:2)
        @test dft4 == dft
    end
    @testset "DataFrame with no missing values and rng" begin
        df = DataFrame(; a=[ 1, 2 ], b=[ 3.0, 4.0 ])
        dft = deepcopy(df)
        rng = StableRNG(1)
        initializemice!(rng, 5, dft)
        @test dft == df
    end
    @testset "DataFrame with missing values in first column, with rng" begin
        df = DataFrame(; a=[ 1, missing, 1 ], b=[ 3.0, 4.0, 5.0 ])
        dft = deepcopy(df)
        rng = StableRNG(1)
        initializemice!(rng, 5, dft)
        # convert missing values in first column
        @test dft == DataFrame(; a=[ 1, MiceValue(5, ones(Int, 5)), 1 ], b=[ 3.0, 4.0, 5.0 ]) 
        # accept rng and String
        dft2 = deepcopy(df)
        initializemice!(rng, 5, dft2, "a")
        @test dft2 == dft
        # accept rng and vector of Strings
        dft3 = deepcopy(df)
        initializemice!(rng, 5, dft3, [ "a", "b" ])
        @test dft3 == dft
        # accept rng and Symbol
        dft4 = deepcopy(df)
        initializemice!(rng, 5, dft4, :a)
        @test dft4 == dft
        # accept rng and vector of Symbols
        dft5 = deepcopy(df)
        initializemice!(rng, 5, dft5, [ :a, :b ])
        @test dft5 == dft
        # accept rng and Index
        dft6 = deepcopy(df)
        initializemice!(rng, 5, dft6, 1)
        @test dft6 == dft
        # accept rng and vector of Indexes
        dft7 = deepcopy(df)
        initializemice!(rng, 5, dft7, 1:2)
        @test dft7 == dft
    end
    @testset "Same rng gives same results" begin
        @testset for i ∈ [ "a", [ "a", "b" ], :a, [ :a, :b ], 1, 1:2 ]
            df = DataFrame(; a=[ 1, missing, 2 ], b=[ 3.0, 4.0, 5.0 ])
            dft_a = deepcopy(df)
            rng_a = StableRNG(1)
            initializemice!(rng_a, 5, dft_a, i)
            dft_b = deepcopy(df)
            rng_b = StableRNG(1)
            initializemice!(rng_b, 5, dft_b, i) 
            @test dft_a == dft_b
        end
    end
    @testset "Different rng gives different results" begin
        @testset for i ∈ [ "a", [ "a", "b" ], :a, [ :a, :b ], 1, 1:2 ]
            df = DataFrame(; a=[ 1, missing, 2 ], b=[ 3.0, 4.0, 5.0 ])
            dft_a = deepcopy(df)
            rng_a = StableRNG(1)
            initializemice!(rng_a, 5, dft_a, i)
            dft_b = deepcopy(df)
            rng_b = StableRNG(2)
            initializemice!(rng_b, 5, dft_b, i) 
            @test dft_a != dft_b
        end
    end
end  
@testset "Manipulate MiceValues" begin
    v = [ 1.0, MiceValue(5, [ 1.0, 2.0, 3.0, 4.0, 5.0 ]), 5.0 ]
    @test micevalues(v, 1) == [ 1.0, 1.0, 5.0 ]
    @test micevalues(v, 2) == [ 1.0, 2.0, 5.0 ]
    df = DataFrame( ; 
        a=[ 1.0, MiceValue(5, [ 1.0, 2.0, 3.0, 4.0, 5.0 ]), 2.0 ], b=[ 3.0, 4.0, 5.0 ]
    )
    @test micevalues(df, 1) == [
        1.0  3.0
        1.0  4.0
        2.0  5.0
    ]
    @test micevalues(df, 2) == [
        1.0  3.0
        2.0  4.0
        2.0  5.0
    ]
    @test micevalues(df, 1, 3) == [
        1.0  
        3.0  
        2.0  
    ]
    @test micevalues(df, "a", 4) == [
        1.0  
        4.0  
        2.0  
    ]
    @test micevalues(df, :a, 5) == [
        1.0  
        5.0  
        2.0  
    ]
    @test micevalues(df, [ :a, :b ], 1) == [
        1.0  3.0
        1.0  4.0
        2.0  5.0
    ]
    @test micevalues(df, 1:2, 2) == [
        1.0  3.0
        2.0  4.0
        2.0  5.0
    ]
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
@testset "Update MiceValues, multiple predictive variables" begin
    df = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [0.5, 5.0, MiceValue(2, [ 0.5, 5.0 ]), 4.0], 
        c = [2.0, MiceValue(2, [ 2.0, 2.0 ]), 10.9, 12.0]
    )
    dfa = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [0.5, 5.0, MiceValue(2, [ 0.9859979256186087, 3.441250555637871 ]), 4.0], 
        c = [2.0, MiceValue(2, [ 2.0, 2.0 ]), 10.9, 12.0]
    )
    df2 = deepcopy(df)
    updatemicevalues!(df2, :b, [ :a, :c ], 2)
    @test df2 == dfa    
    df3 = deepcopy(df)
    updatemicevalues!(df3, "b", [ "a", "c" ], 2)
    @test df3 == dfa
    df4 = deepcopy(df)
    updatemicevalues!(df4, 2, [ 1, 3 ], 2)
    @test df4 == dfa
end
@testset "Update MiceValues, one predictive variable" begin
    df = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [0.5, 5.0, MiceValue(2, [ 0.5, 5.0 ]), 4.0], 
        c = [2.0, MiceValue(2, [ 2.0, 2.0 ]), 10.9, 12.0]
    )
    dfa = DataFrame(
        a = [ 1.0, 2.0, 3.0, 4.0 ], 
        b = [0.5, 5.0, MiceValue(2, [ 0.9859979256186087, 3.441250555637871 ]), 4.0], 
        c = [2.0, MiceValue(2, [ 2.0, 2.0 ]), 10.9, 12.0]
    )
    df2 = deepcopy(df)
    updatemicevalues!(df2, :b, :a, 2)
    @test df2 == dfa    
    df3 = deepcopy(df)
    updatemicevalues!(df3, "b", [ "a", "c" ], 2)
    @test df3 == dfa
    df4 = deepcopy(df)
    updatemicevalues!(df4, 2, [ 1, 3 ], 2)
    @test df4 == dfa
end
end  # @testset "SimpleMice.jl"
