using Test
using Statistics
using Random
using TrackerFlux
using Tracker
using Flux
using Pkg
using UUIDs

Random.seed!(1234)

@testset "TrackerFlux.jl" begin

x = randn(Float32, 10, 1, 100)
y = mean(x, dims = 1)

θ0, re = Flux.destructure(Dense(2, 1))
@test Tracker.gradient(θ0) do θ
    sum(re(θ)([0.5, 2.0]))
end[1].data == [0.5, 2.0, 1]

model = Chain(LSTM(10, 100), LSTM(100, 1)) |> TrackerFlux.track
θ0, re = Flux.destructure(model)
@test Flux.destructure(re(θ0))[1] == θ0

model(x[:, :, 1])
@test Tracker.istracked(model[1].state[1])
Flux.truncate!(model)
ver = Pkg.dependencies()[UUID("587475ba-b771-5e3f-ad9e-33799f191a9c")].version
ver < v"0.12" && @test !Tracker.istracked(model[1].state[1])
@test !Tracker.istracked(TrackerFlux.untrack(model)[1].cell.Wh)

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
@test loss(x, y) < 0.02

end