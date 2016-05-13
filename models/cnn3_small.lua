require 'nn'

local model = nn.Sequential()

--model:add(nn.View(64*48*3))
--model:add(nn.Dropout(0.5))
model:add(nn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.Dropout(0.2))
model:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.Dropout(0.3))
model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.Dropout(0.4))
model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.Dropout(0.4))
print( model:forward(torch.FloatTensor(1,3,48,64)):size())
local n = 256*32*24/16/24
model:add(nn.View(n))
model:add(nn.Linear(n, 128))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.5))
model:add(nn.Tanh())
model:add(nn.Linear(128, 10))
model:add(nn.SoftMax())

--[[initialization from MSR]]
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

--MSRinit(model)



return model
