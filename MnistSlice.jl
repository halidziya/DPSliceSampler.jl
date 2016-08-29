using Plots
using Munkres
using MLBase
include("Slice.jl")

function readMat(filename)
  fs = open(filename,"r")
  n  = read(fs,Int32)
  d  = read(fs,Int32)
  data = zeros(d,n)
  read!(fs,data)
  close(fs)
  return data
end

data = readMat("mnist.matrix")
y = ceil(Int32,(1:60000)/6000)
scatter(data[1,:],data[2,:],color=y)

@time (z,μ,Σ,β) = SliceSampler(data)

xt = data';
plot(scatter(xt[:,1],xt[:,2],color=z,marker=(:+,2)),scatter(xt[:,1],xt[:,2],color=y,marker=(:+,2)))
include("HalidUtils.jl")
plotCovs(μ[β.>0.01],Σ[β.>0.01])
plot!()

cost = confusmat(10,y,z)
cost = sum(cost,2) .- cost
zmap = munkres(cost)
zmap[zmap] = 1:10
cm = confusmat(10,y,zmap[z])
ts = sum(cm,2)
ps = sum(cm,1)
prec = diag(cm)./ps'
recall = diag(cm)./ts
print mean(2./(1./prec + 1./recall))
