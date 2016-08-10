require 'loadcaffe'
require 'cudnn'
require 'nn'
require 'cunn'
require 'matt'


model = loadcaffe.load('./models/caffemodels/VGG_ILSVRC_16_layers_deploy.prototxt', './models/caffemodels/VGG_ILSVRC_16_layers.caffemodel', 'cudnn')
model:remove(40)
model:remove(39)

model:add(nn.Linear(4096,10))
model:add(nn.SoftMax())

print (model)
model = model:cuda()
return model


