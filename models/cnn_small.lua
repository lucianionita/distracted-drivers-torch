require 'nn'

local model = nn.Sequential()

--model:add(nn.View(64*48*3))
--model:add(nn.Dropout(0.5))
model:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, 1, 1))
model:add(nn.SpatialMaxPooling(2,2,2,2))
--print (model:forward(torch.FloatTensor(1,3,48,64):zero()))
local n = 32*32*24/4/4
model:add(nn.View(n))
model:add(nn.Linear(n, 10))
model:add(nn.SoftMax())

--[[initialization from MSR
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

MSRinit(vgg)
--]]


return model
