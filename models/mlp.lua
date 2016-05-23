require 'nn'

local model = nn.Sequential()

model:add(nn.View(64*48*3))
--model:add(nn.Dropout(0.5))
model:add(nn.Linear(64*48*3, 1000))
model:add(nn.Tanh())
model:add(nn.Linear(1000,10))
model:add(nn.SoftMax())


return model
