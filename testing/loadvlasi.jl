
import Vlasiator as vs

meta = vs.load(ENV["VLSV_ASSETS_PATH"]*"/bulk1.0001280.vlsv")
#meta = vs.load("bulk10.0000000.vlsv")

cid = 356780649

species="proton"
vs.init_cellswithVDF!(meta, "proton")
locations = [vs.getcellcoordinates(meta, cid) for cid in meta.meshes[species].cellwithVDF]

cid = if cid == -1
  cid = vs.getcell(meta, locations[20])
else 
  cid
end

if false
_, N = [
  let cid = vs.getcell(meta, locations[k])
    try
      id, _ = vs.readvcells(meta,cid)
      length(id)
    catch e 
      0 
    end
  end 
  for k in axes(locations,1) ] |> findmax
end

# cid = vs.getcell(meta, locations[N])

id, vf = vs.readvcells(meta, cid)

projected = hcat((vs.getvcellcoordinates(meta,id) ./ 40000 .|> x->x[1:2])...) .|>ceil .|> Int64
projected_float = hcat((vs.getvcellcoordinates(meta,id) ./ 80000 .|> x->x[1:2])...)
full3d = hcat((vs.getvcellcoordinates(meta,id) ./ 40000 .|> x->x[1:3])...) .|> ceil .|> Int64

limits3d = [
  minimum(full3d[1,:]) maximum(full3d[1,:]);
  minimum(full3d[2,:]) maximum(full3d[2,:]); 
  minimum(full3d[3,:]) maximum(full3d[3,:]) ]

limits = [ 
  minimum(projected[1,:]) maximum(projected[1,:]);
  minimum(projected[2,:]) maximum(projected[2,:]) ]

vdf_image = zeros((limits[1,2]-limits[1,1]+1, limits[2,2]-limits[2,1]+1) .|>Int64)
vdf_image3 = zeros((limits3d[1,2]-limits3d[1,1]+1, limits3d[2,2]-limits3d[2,1]+1, limits3d[3,2]-limits3d[3,1]+1) .|>Int64)

bias = [limits[1,1], limits[2,1]] .-1
bias3 = [limits3d[1,1], limits3d[2,1], limits3d[3,1]] .-1
scaler = maximum(vf)
scaler = typeof(vf[1])(1)
for d in axes(vf,1)
  v = projected[:,d]
  v3 = full3d[:,d]
  ind = v - bias
  ind3 = v3 - bias3
  vdf_image[ind[1],ind[2]] = vdf_image[ind[1],ind[2]]+vf[d]/scaler
  #vdf_image3[ind3[1],ind3[2],ind3[3]] = vdf_image3[ind3[1], ind3[2], ind3[3]] + v3[d]/scaler

  vdf_image3[ind3...] = vdf_image3[ind3...] + vf[d]/scaler
end

#image(vdf_image)
