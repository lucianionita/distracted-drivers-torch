require 'nn'

local model = nn.Sequential()

--model:add(nn.View(64*48*3))
--model:add(nn.Dropout(0.5))
model:add(nn.SpatialConvolution(3, 16, 11, 11, 1, 1, 5, 5) 
model:add(nn.View(16*64*48))
model:add(nn.Linear(16 * (64*48), 10))
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
