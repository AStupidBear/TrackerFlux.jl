using Test
using Statistics
using TrackerFlux
using Tracker
using Flux

@testset "TrackerFlux.jl" begin

x = randn(Float32, 10, 1, 100)
y = mean(x, dims = 1)

model = Chain(LSTM(10, 100), LSTM(100, 1)) |> TrackerFlux.track
θ, re = Flux.destructure(model)
@test Flux.destructure(re(θ))[1] == θ

model(x[:, :, 1])
@test Tracker.istracked(model[1].state[1])
Flux.truncate!(model)
@test !Tracker.istracked(model[1].state[1])

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

@test loss(x, y) < 0.02

end