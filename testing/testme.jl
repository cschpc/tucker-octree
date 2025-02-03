import Libdl
using FFTW

module TOctreeCompress

import Libdl

libbi=Libdl.dlopen("libtoctree_compressor.so")
compress = Libdl.dlsym(libbi, :compress_with_toctree_method)
compress_2d = Libdl.dlsym(libbi, :compress_with_toctree_method_2d)
uncompress = Libdl.dlsym(libbi, :uncompress_with_toctree_method)
uncompress_2d = Libdl.dlsym(libbi, :uncompress_with_toctree_method_2d)


function compress!(data::Array{Cfloat,3}, tol;maxiter=100, skip_levels=6)::Vector{UInt8}

  c_Nx = Csize_t(size(data,1))
  c_Ny = Csize_t(size(data,2))
  c_Nz = Csize_t(size(data,3))
  c_tol = Cfloat(tol)

  c_bytes = Vector{Ptr{UInt8}}(undef,1)
  c_n_bytes = Vector{UInt64}(undef,1)

  @ccall $compress(
    data::Ptr{Cfloat}, c_Nx::Csize_t, c_Ny::Csize_t, c_Nz::Csize_t, 
    c_tol::Cfloat, c_bytes::Ptr{Ptr{UInt8}}, c_n_bytes::Ptr{UInt64},maxiter::UInt64, skip_levels::UInt64)::Cvoid

  n_bytes = c_n_bytes[1]
  unsafe_bytes = unsafe_wrap(Array{Cuchar,1}, c_bytes[1], n_bytes; own = true)
  bytes = similar(unsafe_bytes)
  bytes .= unsafe_bytes
  return bytes
end

function compress!(data::Array{Cfloat,2}, tol;maxiter=100, skip_levels=6)::Vector{UInt8}

  c_Nx = Csize_t(size(data,1))
  c_Ny = Csize_t(size(data,2))
  c_tol = Cfloat(tol)

  c_bytes = Vector{Ptr{UInt8}}(undef,1)
  c_n_bytes = Vector{UInt64}(undef,1)

  @ccall $compress_2d(
    data::Ptr{Cfloat}, c_Nx::Csize_t, c_Ny::Csize_t,
    c_tol::Cfloat, c_bytes::Ptr{Ptr{UInt8}}, c_n_bytes::Ptr{UInt64},maxiter::UInt64, skip_levels::UInt64)::Cvoid

  n_bytes = c_n_bytes[1]
  unsafe_bytes = unsafe_wrap(Array{Cuchar,1}, c_bytes[1], n_bytes; own = true)
  bytes = similar(unsafe_bytes)
  bytes .= unsafe_bytes
  return bytes
end

function uncompress!(data::Array{Cfloat,3}, bytes::Vector{UInt8})
  n_bytes = UInt64(size(bytes,1))
  c_Nx = Csize_t(size(data,1))
  c_Ny = Csize_t(size(data,2))
  c_Nz = Csize_t(size(data,3))
  @ccall $uncompress(
    data::Ptr{Float32}, c_Nx::Csize_t, c_Ny::Csize_t, c_Nz::Csize_t, 
    bytes::Ptr{UInt8}, n_bytes::UInt64)::Cvoid
end

function uncompress!(data::Array{Cfloat,2}, bytes::Vector{UInt8})
  n_bytes = UInt64(size(bytes,1))
  c_Nx = Csize_t(size(data,1))
  c_Ny = Csize_t(size(data,2))
  c_Nz = Csize_t(size(data,3))
  @ccall $uncompress_2d(
    data::Ptr{Float32}, c_Nx::Csize_t, c_Ny::Csize_t,
    bytes::Ptr{UInt8}, n_bytes::UInt64)::Cvoid
end

end

# testing

if true
  import .TOctreeCompress as to
  Nx = Csize_t(50)
  Ny = Csize_t(50)
  Nz = Csize_t(50)
  tol = Cfloat(0.05)

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

  @time bytes = to.compress!(data, tol;maxiter=1024, skip_levels=0)

  println("residual max: ", maximum(abs.(data)))

  undata = zeros(Cfloat, size(data)) 
  undata .= Cfloat(0)

  println("Ratio: $((size(bytes,1)/(4*(size(data)|>prod)))^(-1))")

  @time to.uncompress!(undata, bytes)

  println("undata-origdata absmax: ", maximum(abs.(origdata.-undata)))
  ranges = (1:25, 1:2,1)
  display(undata[ranges...])
  display(origdata[ranges...])
Libdl.dlclose(to.libbi)
end

# fig = Figure(); 
if false
using TestImages
using GLMakie
using FFTW
using TiffImages

import .TOctreeCompress as to

# img = testimage("fabio_gray_512") .|> Float32
# img = testimage("cameraman") .|> Float32
# img = testimage("brick_wall_he_512.tiff") .|> Float32
# img = testimage("livingroom.tif") .|> Float32
img = TiffImages.load("./blobby-nsq.tiff") .|> Float32

# img = TiffImages.load("blobby-nsq-2level-big.tiff") .|> Float32



fig = Figure(); 
# image!(fig[1,1], rotr90(img))#, axis = (aspect=DataAspect(),))

image(fig[1,1], rotr90(img), axis = (aspect=DataAspect(),))

# img = dct(img, [1,2])

mult = let C=1.1, Cx = 1.0*C / size(img,1), Cy = 1.0*C / size(img,2)
  [sqrt(1/(img[1,1]^2)+(Cx*(x-1))^2+(Cy*(y-1))^2) for x in axes(img,1), y in axes(img, 2)]
  end

# img = img .* mult

logtrx(x;eps=1.8, beta=2) = log(beta*x+eps)
invlogtrx(y; eps=1.8, beta=2) = (exp(y) - eps)/beta # y= log(x*beta+eps) x*beta+eps = log(y) x = (log(y)-eps)/beta

parfrac(x;eps=0.1) = 1/(x+eps)
invparfrac(y;eps=0.1) = 1/y - eps # y = 1/(x+eps) 1/y = x+eps x = 1/y - eps

aff(x;eps=2) = x+eps
invaff(x;eps=2) = x-eps

id(x) = x

F = id; #x->x
invF = id; #F

normalize(X) = (X .- minimum(X)) ./ (maximum(X) - minimum(X))

img2 = F.(copy(img)) .|> Float32; 

iters=1
# maxiters=[maxiters[1] + 1]
maxiters = [4096]
println("maxiters = $(maxiters[1])")

skip_levels = UInt64(floor(log2(maximum(size(img))/8)))
skip_levels=4
bytes = [to.compress!(img2, 5e-2; maxiter=maxiters[n], skip_levels=skip_levels) for n = 1:iters]

println("Ratio: $(prod(size(img))/sum(size.(bytes,1)))")

acc = copy(img2);
# img2[:] .= Float32(0.0)

acc[:] .= Float32(0.0)

for n = iters:-1:1
  to.uncompress!(img2, bytes[n])
  acc[:] = acc[:] .+ img2[:]
end

# acc = acc ./ mult
# acc = idct(acc, [1,2])

print(sum(isnan.(acc) .| isinf.(acc)))
acc[isnan.(acc) .| isinf.(acc) .| (acc .< 0)] .= Float32(0)

# acc = acc .- sum(acc)/prod(size(acc)) .+ sum(img)/(prod(size(img)))
image(fig[1,2], invF.(acc)|>rotr90, axis=(aspect=DataAspect(),) );
# image!(fig[1,2], invF.(acc)|>rotr90)#, axis=(aspect=DataAspect(),) );
Libdl.dlclose(to.libbi)

fig

end
