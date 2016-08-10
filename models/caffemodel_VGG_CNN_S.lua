require 'loadcaffe'
require 'cudnn'
require 'nn'



model = loadcaffe.load('./models/caffemodels/VGG_CNN_S_deploy.prototxt', './models/caffemodels/VGG_CNN_S.caffemodel')
model:get(22):reset()
model:get(19):reset()
--model:get(16):reset()
model:remove(23)
model:remove(22)
model:remove(21)
model:remove(20)
model:remove(19)

model:add(nn.Linear(4096,10))
model:add(nn.SoftMax())

print (model)
model = model:cuda()
return model


