import Libdl

module TOctreeCompress

import Libdl

libbi=Libdl.dlopen("libtoctree_compressor.so")
compress = Libdl.dlsym(libbi, :compress_with_toctree_method)
uncompress = Libdl.dlsym(libbi, :uncompress_with_toctree_method)


function toctree_compress!(data::Array{Cfloat,3}, tol;maxiter=100)::Vector{UInt8}

  c_Nx = Csize_t(size(data,1))
  c_Ny = Csize_t(size(data,2))
  c_Nz = Csize_t(size(data,3))
  c_tol = Cfloat(tol)

  c_bytes = Vector{Ptr{UInt8}}(undef,1)
  c_n_bytes = Vector{UInt64}(undef,1)

  @ccall $compress(
    data::Ptr{Cfloat}, c_Nx::Csize_t, c_Ny::Csize_t, c_Nz::Csize_t, 
    c_tol::Cfloat, c_bytes::Ptr{Ptr{UInt8}}, c_n_bytes::Ptr{UInt64},maxiter::UInt64)::Cvoid

  n_bytes = c_n_bytes[1]
  unsafe_bytes = unsafe_wrap(Array{Cuchar,1}, c_bytes[1], n_bytes; own = true)
  bytes = similar(unsafe_bytes)
  bytes .= unsafe_bytes
  return bytes
end

function toctree_uncompress!(data::Array{Cfloat,3}, bytes::Vector{UInt8})
  n_bytes = UInt64(size(bytes,1))
  c_Nx = Csize_t(size(data,1))
  c_Ny = Csize_t(size(data,2))
  c_Nz = Csize_t(size(data,3))
  @ccall $uncompress(
    data::Ptr{Float32}, c_Nx::Csize_t, c_Ny::Csize_t, c_Nz::Csize_t, 
    bytes::Ptr{UInt8}, n_bytes::UInt64)::Cvoid
end

end

# testing


Nx = Csize_t(50)
Ny = Csize_t(50)
Nz = Csize_t(50)
tol = Cfloat(0.01)

# data = Vector{Cfloat}(undef, Nx*Ny*Nz)
data = Array{Cfloat,3}(undef, Nx, Ny, Nz)
origdata = similar(data)

for i in axes(data,1)
  for j in axes(data,2)
    for k in axes(data,3)
      data[i,j,k] = exp(-0.1*(i+j+k)^2/(Nx+Ny+Nz))*cos(8*pi*Cfloat(i+2+j+k)/(Nx+Ny+Nz))
      origdata[i,j,k] = exp(-0.1*(i+j+k)^2/(Nx+Ny+Nz))*cos(8*pi*Cfloat(i+2+j+k)/(Nx+Ny+Nz))
    end
  end
end

println("data max: ", maximum(abs.(data)))
import .TOctreeCompress as to

@time bytes = to.toctree_compress!(data, tol;maxiter=10)


println("residual max: ", maximum(abs.(data)))

undata = zeros(Cfloat, size(data)) 
undata .= Cfloat(0)

@time to.toctree_uncompress!(undata, bytes)

println("undata-origdata absmax: ", maximum(abs.(origdata.-undata)))
ranges = (1:25, 1:2,1)
display(undata[ranges...])
display(origdata[ranges...])
