 
googlenet = dofile('./models/inception.torch/googlenet.lua')
net = googlenet({
  cudnn.SpatialConvolution,
  cudnn.SpatialMaxPooling,
  cudnn.ReLU,
  cudnn.SpatialCrossMapLRN
})
net:cuda()

net:remove(26)
net:remove(25)
net:remove(24)

net:add(nn.Linear(1024,10))
net:add(cudnn.SoftMax())



print (net)
return net


