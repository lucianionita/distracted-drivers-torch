require 'provider'
require 'xlua'
require 'optim'
require 'nn'
require 'dataloader'
c = require 'trepl.colorize'


-- Create submission file
-------------------------

function create_submission(model_files)

    print (c.red"Creating submission. Data will be discarded.")
    provider.data = nil
    provider.labels = nil


    predictions = torch.Tensor(provider.test:size(1),10)
    predictions[{}] = 1
    predictions = predictions:cuda()

for k,model_file in pairs(model_files) do


	collectgarbage()
	print ("Loading model " .. model_file)
	model = torch.load(model_file);

    for idx = 1,provider.test:size(1),16 do

        hi_idx = math.min(idx + 16-1, provider.test:size(1))

        -- update progress
        xlua.progress(idx, provider.test:size(1))

        -- set up batch
        local inputs = provider.test[{{idx, hi_idx},{}}]
        bs = hi_idx - idx + 1

        -- get the outputs
        local agg_outputs = torch.Tensor(bs, 10):cuda()
		agg_outputs[{}]=0

--        for i, trainer in pairs(trainers) do

                local outputs = model:forward(inputs)
				outputs[outputs:lt(1e-5)] = 1e-5

        if bs == 1 then
            outputs = outputs:reshape(1, outputs:size(1))
        end
                predictions[{{idx, hi_idx},{}}] =  predictions[{{idx, hi_idx},{}}] + outputs
--		--agg_outputs = agg_outputs + outputs
--        end
	
--	agg_outputs = agg_outputs / #trainers

        -- fix for batchsize 1 
--        if bs == 1 then
--            agg_outputs = agg_outputs:reshape(1, agg_outputs:size(1))
--        end
--        predictions[{{idx, hi_idx},{}}] = agg_outputs
    end
end

	predictions = predictions / #model_files

	

	-- Write the file submission.csv

	f = torch.DiskFile("submission.csv", "w")
	f:writeString("img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n")
	for idx = 1,predictions:size(1) do
		file_name = provider.test_files[idx]
		pos = file_name:find("img_")
		file_name = file_name:sub(pos, file_name:len())	
		f:writeString("" .. file_name .. ",")
		for j = 1,10 do
			f:writeString(("%.6f"):format(predictions[idx][j]))
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


provider = torch.load('p224test.t7')

model_files = {"saved_models/p1_10.mdl",
"saved_models/p1_11.mdl",
"saved_models/p1_12.mdl",
"saved_models/p1_13.mdl",
"saved_models/p1_14.mdl",
"saved_models/p1_15.mdl",
"saved_models/p1_16.mdl",
"saved_models/p1_17.mdl",
"saved_models/p1_18.mdl",
"saved_models/p1_19.mdl",
"saved_models/p1_1.mdl",
"saved_models/p1_20.mdl",
"saved_models/p1_21.mdl",
"saved_models/p1_22.mdl",
"saved_models/p1_23.mdl",
"saved_models/p1_24.mdl",
"saved_models/p1_25.mdl",
"saved_models/p1_26.mdl",
"saved_models/p1_2.mdl",
"saved_models/p1_3.mdl",
"saved_models/p1_4.mdl",
"saved_models/p1_5.mdl",
"saved_models/p1_6.mdl",
"saved_models/p1_8.mdl",
"saved_models/p1_9.mdl"
}

create_submission (model_files)

