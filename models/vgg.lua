require 'nn'

local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane, sx, sy)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, sx,sy, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-5))
  vgg:add(nn.ReLU(true))
  return vgg
end



-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling

ConvBNReLU(3,32, 1, 1)
vgg:add(nn.Dropout(0.5))

ConvBNReLU(32,32, 1, 1)
vgg:add(nn.Dropout(0.5))

ConvBNReLU(32,64, 2, 2)
vgg:add(nn.Dropout(0.5))

ConvBNReLU(64,64, 1, 1)
vgg:add(nn.Dropout(0.5))

ConvBNReLU(64,128, 2, 2)
vgg:add(nn.Dropout(0.5))

ConvBNReLU(128,128, 1, 1)
vgg:add(nn.Dropout(0.5))

vgg:add(nn.View(128*64*48/4/4))
vgg:add(nn.Linear(128*64*48/4/4, 128))
vgg:add(nn.Tanh())
vgg:add(nn.Dropout(0.5))
vgg:add(nn.Linear(128, 10))
vgg:add(nn.SoftMax())


-- initialization from MSR
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

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return vgg
