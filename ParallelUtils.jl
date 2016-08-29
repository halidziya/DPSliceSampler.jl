if (nprocs()==1)
  addprocs(8)
end
# This function retuns the (irange,jrange) indexes assigned to this worker
@everywhere function chunk(q::SharedArray)
    idx = indexpids(q)
    if idx == 0
      return 1:0
    end
    nchunks = length(procs(q))
    splits = [round(Int, s) for s in linspace(0,size(q,2),nchunks+1)]
    splits[idx]+1:splits[idx+1]
end

@everywhere function applyColumns(q,jrange,f)
    for j in jrange
        q[:,j] = f(q[:,j])
    end
end

@everywhere function reduceColumns(r,q,jrange,f)
    for j in jrange
        r[j] = f(q[:,j])
    end
end




@everywhere function writeColumns(q,jrange,val)
    for j in jrange
        q[:,j] = val
    end
end

@everywhere applyColumns(q,f) = applyColumns(q, chunk(q),f)
@everywhere reduceColumns(r,q,f) = reduceColumns(r , q, chunk(q),f)
@everywhere writeColumns(q,val) = writeColumns(q, chunk(q),val)

function applyComlumnShared(q,f)
    @sync begin
        for p in procs(q)
            @async remotecall_wait(applyColumns,p,q,f)
        end
    end
end

function reduceComlumnShared(r,q,f)
    @sync begin
        for p in procs(q)
            @async remotecall_wait(reduceColumns,p,r,q,f)
        end
    end
end


function sharedFill(q::SharedArray,val)
  @sync begin
      for p in procs(q)
          @async remotecall_wait(writeColumns,p,q,val)
      end
  end
end
