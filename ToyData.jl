@everywhere using Distributions
include("Slice.jl")

# Generate Toy Data
const D = 2
const NCOMP = 5
const NPOINTS = 1000
const μ₀ = zeros(D)

mvn = MultivariateNormal(μ₀,eye(D))
μt  = [rand(mvn) for i=1:NCOMP]
wsh = InverseWishart(D+2,eye(D)/10)
Σt =[rand(wsh) for i=1:NCOMP]
dr = Dirichlet(ones(NCOMP))
wᵢ = rand(dr)

mixdistribution  = Multinomial(1,wᵢ)
mvns =  [MultivariateNormal(μt[i],Σt[i]) for i=1:NCOMP]
x = SharedArray(Float64,(D,NPOINTS))
y = zeros(Int32,NPOINTS)
for i=1:NPOINTS
  y[i] = find(rand(mixdistribution))[1]
  x[:,i] = rand(mvns[y[i]])
end
#x = x[:,sortperm(y)]
#y = y[sortperm(y)]

(z,μ,Σ,β) = SliceSampler(x)

xt = x';
plot(scatter(xt[:,1],xt[:,2],color=z,marker=(:+,2)),scatter(xt[:,1],xt[:,2],color=y,marker=(:+,2)))
include("HalidUtils.jl")
plotCovs(μ[β.>0.01],Σ[β.>0.01])
plot!()
