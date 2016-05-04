-- Configure the model
------------------------------------
function get_trainer()

    -- configure model  
    model = nn.Sequential()
    model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
    model:add(cast(dofile('models/'..opt.model..'.lua')))
    model:get(2).updateGradInput = function(input) return end

    -- cast to cudnn if necessary
    if opt.backend == 'cudnn' then
       cudnn.convert(model:get(2), cudnn)
    end

    -- get the parameters and gradients
    parameters, gradParameters = model:getParameters()
    parameters = cast(parameters)
    gradParameters = cast(gradParameters)

    -- create the criterion
    criterion = nn.CrossEntropyCriterion()
    criterion = criterion:float()
    criterion = cast(criterion)

    -- set up the optimizer parameters
    optimState = {
        learningRate = opt.learningRate,
        weightDecay = opt.weightDecay,
        momentum = opt.momentum,
        learningRateDecay = opt.learningRateDecay,
    }

    trainer = {}

    trainer.model = model
    trainer.params = parameters
    trainer.gParams = gradParameters
    trainer.criterion = criterion
    trainer.optimState = optimState

    return trainer
end

