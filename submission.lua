
-- Create submission file
-------------------------

function create_submission(trainers)

    print (c.red"Creating submission. Data will be deleted.")
    provider.data = nil
    provider.labels = nil

    print ("img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9")
    predictions = torch.Tensor(provider.test:size())
    predictions[{}] = 1
    for idx = 1,provider.test:size(1),opt.batchSize do

        hi_idx = math.max(idx + opt.batchSize-1, provider.test:size(1))

        -- update progress
        xlua.progress(t, #indices)

        -- set up batch
        local inputs = provider.test[{{idx, hi_idx},{}}]
        bs = hi_idx - idx + 1

        -- get the outputs
        local agg_outputs = torch.Tensor(bs, 10)
        for i, trainer in pairs(trainers) do

                local outputs = trainer.model:forward(inputs)
                agg_outputs = agg_outputs * outputs

        end
        -- geometric mean
        agg_outputs = torch.pow(outputs, #trainer)

        -- fix for batchsize 1 
        if bs == 1 then
            agg_outputs = agg_outputs:reshape(1, agg_outputs:size(1))
        end

        predictions[{{idx, hi_idx},{}}] = agg_outputs
    end
end


