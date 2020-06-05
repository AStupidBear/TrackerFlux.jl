# TrackerFlux

[![Build Status](https://travis-ci.com/AStupidBear/TrackerFlux.jl.svg?branch=master)](https://travis-ci.com/AStupidBear/TrackerFlux.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/AStupidBear/TrackerFlux.jl?svg=true)](https://ci.appveyor.com/project/AStupidBear/TrackerFlux-jl)
[![Coverage](https://codecov.io/gh/AStupidBear/TrackerFlux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/AStupidBear/TrackerFlux.jl)

## Example

```julia
using Statistics
using TrackerFlux
using Flux

x = randn(Float32, 10, 1, 100)
y = mean(x, dims = 1)

model = Chain(LSTM(10, 100), LSTM(100, 1)) |> TrackerFlux.track

function loss(x, y)
    xs = Flux.unstack(x, 3)
    ys = Flux.unstack(y, 3)
    ŷs = model.(xs)
    l = 0f0
    for t in 1:length(ŷs)
        l += Flux.mse(ys[t], ŷs[t])
    end
    return l / length(ŷs)
end

opt = ADAMW(1e-3, (0.9, 0.999), 1e-4)

cb = () -> Flux.reset!(model)

Flux.@epochs 10 Flux.train!(loss, params(model), repeat([(x, y)], 100), opt, cb = cb)
```

Note that `TrackerFlux` will overload `Flux.Zygote.gradient` to avoid repetitive definition of `Flux.train!` therefore you cannot use `Zygote.gradient` after importing `TrackerFlux`.