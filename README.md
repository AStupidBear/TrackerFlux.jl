# TrackerFlux

[![Build Status](https://github.com/AStupidBear/TrackerFlux.jl/workflows/CI/badge.svg)](https://github.com/AStupidBear/TrackerFlux.jl/actions)
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
ps = Flux.params(model)
data = repeat([(x, y)], 100)
opt = ADAMW(1e-3, (0.9, 0.999), 1e-4)
cb = () -> Flux.reset!(model)
TrackerFlux.overload_gradient()
Flux.@epochs 10 Flux.train!(loss, ps, data, opt, cb = cb)
```
