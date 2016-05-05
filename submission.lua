
-- Create submission file
-------------------------

function create_submission(trainers)

    print (c.red"Creating submission. Data will be discarded.")
    provider.data = nil
    provider.labels = nil

    predictions = torch.Tensor(provider.test:size(1),10)
    predictions[{}] = 1
    for idx = 1,provider.test:size(1),opt.batchSize do

        hi_idx = math.min(idx + opt.batchSize-1, provider.test:size(1))

        -- update progress
        --xlua.progress(idx, provider.test:size(1))

        -- set up batch
        local inputs = provider.test[{{idx, hi_idx},{}}]
        bs = hi_idx - idx + 1

        -- get the outputs
        local agg_outputs = torch.Tensor(bs, 10)
		agg_outputs[{}]=1
        for i, trainer in pairs(trainers) do

                local outputs = trainer.model:forward(inputs)
                agg_outputs = torch.cmul(agg_outputs, outputs)
				--print (i,outputs:transpose(1,2))
				--print (i,agg_outputs:transpose(1,2))
        end
        -- geometric mean
        agg_outputs = torch.pow(agg_outputs, 1/#trainers)

        -- fix for batchsize 1 
        if bs == 1 then
            agg_outputs = agg_outputs:reshape(1, agg_outputs:size(1))
        end
        predictions[{{idx, hi_idx},{}}] = agg_outputs
    end


	-- Write the file submission.csv

	f = torch.DiskFile("submission.csv", "w")
    f:writeString("img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n")
	for idx = 1,predictions:size(1) do
		file_name = provider.test_files[idx]
		pos = file_name:find("img_")
		print(pos, file_name)
		file_name = file_name:sub(pos, file_name:len())	
		f:writeString("" .. file_name .. ",")
		for j = 1,10 do
			f:writeString("" .. predictions[idx][j])
			if j == 10 then
				f:writeString("\n")
			else
				f:writeString(",")
			end

		end
	end	
	f:close()

	return predictions
end


