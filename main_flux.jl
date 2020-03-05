using Flux
using CuArrays
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using CSV, FileIO
using Images
using BSON
CuArrays.allowscalar(false)

include("unet_flux.jl")


img_train_path, img_test_path = "C:/Users/CCL/unet/TCGA/imgs/train/new/", "C:/Users/CCL/unet/TCGA/imgs/test/"
mask_train_path, mask_test_path = "C:/Users/CCL/unet/TCGA/masks/train/new/", "C:/Users/CCL/unet/TCGA/masks/test/"

train, test = collect(CSV.read("C:/Users/CCL/unet/TCGA/imgs/train.csv", header=["fname"]).fname), collect(CSV.read("C:/Users/CCL/unet/TCGA/imgs/test.csv", header=["fname"]).fname)
train_size, test_size = length(train), length(test)

mb_size = 1
mb_idxs = gpu.(collect(partition(1:train_size, mb_size)))

function load_me(img_path, mask_path, path, idxs, testing)
    y1, y2 = ones(256, 256, 3, 1), ones(256, 256, 1, 1)
    for i in idxs
        x1 = Float32.(permutedims(channelview(load(img_path*path[i])), (2, 3, 1)))
        x1 = reshape(x1, (size(x1)..., 1))

        x2 = testing ? Float32.(channelview(load(mask_path*path[i]))) : (Float32.(channelview(load(mask_path*path[i])))[1, :, :])

        x2 = reshape(x2, (size(x2)..., 1, 1))

        y1 = cat(y1, x1, dims=4)
        y2 = cat(y2, x2, dims=4)
    end
    y1 = y1[:, :, :, 2:length(idxs).+1]
    y2 = y2[:, :, :, 2:length(idxs).+1]
    y = (y1, y2)
    return y
end
load_me = gpu(load_me)

UNet = gpu(UNet)
testset = gpu.(load_me(img_test_path, mask_test_path, test, collect(1:length(test)), true))

loss(x, y) = logitbinarycrossentropy(UNet(x), y)
loss = gpu(loss)

function accuracy(x,y)
    y_hat = UNet(x)
    return 2 * sum(y_hat .* y) / (sum(y_hat) + sum(y))
end
accuracy = gpu(accuracy)

optimiser = ADAM()
best_acc, last_improve, epoch_num, threshold = 0.0, 0, 200, 0.95

for i in 1:epoch_num
    for o in 1:length(mb_idxs)
        train_batch = gpu.(load_me(img_train_path, mask_train_path, train, mb_idxs[o], false))
        Flux.train!(loss, gpu.(params(UNet)), train_batch, optimiser)
    end
    acc = accuracy(testset...)
    if acc > best_acc
        model = cpu(UNet)
        BSON.@save "best_model.BSON" model
        global best_acc = acc
        println("New best accuracy!")
    end
    println("Epoch ", i, ": ", acc, "\n")
end
