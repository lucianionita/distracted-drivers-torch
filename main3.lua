-- To create a new data file, this is what you need to do:
-- TODO add method to create upload submission
--require 'trepl'
--arg = {}
--arg[1] = '--gen_data'
--arg[2] = 'y'
require 'provider'
require 'xlua'
require 'optim'
require 'nn'
require 'provider'
c = require 'trepl.colorize'
torch.setdefaulttensortype('torch.FloatTensor')

-- Parse the command line arguments
--------------------------------------
opt = lapp[[
	--model 	(default linear_logsoftmax) 	model name
	-b,--batchSize 	(default 32) 			batch size
 	-r,--learningRate 	(default 1) 		learning rate
 	--learningRateDecay 	(default 1e-7) 		learning rate decay
	
	-s,--save 	(default "logs") 		subdirectory to save logs
	-S,--submission	(default no)			generate(overwrites) submission.csv file

	-f,--n_folds	(default 3)				number of folds to use
	-g,--gen_data 	(default no) 			whether to generate data file 
	-d,--datafile 	(default p.t7) 			file name of the data provider
	-h,--height	(default 48)			height of the input images
	-w,--width	(default 64)			width of the resized images
	--L2		(default 0)			L2 norm
	--L1		(default 0)			L1 norm

	-t,--trainAlgo	(default sgd)			training algorithm: sgd, adam, 
	--weightDecay 	(default 0.0005) 		weightDecay
	-m,--momentum 	(default 0.9) 			momentum
	--epoch_step 	(default 25) 			epoch step
	--max_epoch 	(default 300) 			maximum number of iterations

 	--backend (default cudnn) 			backend to be used nn/cudnn
 	--type (default cuda) 				cuda/float/cl

	-v,--validation (default 6) 			number of drivers to use in validation set
]]


if opt.backend == 'cudnn' then
	   require 'cudnn'
end

-- Aux function to generate a string form a table of drivers
-------------------------------------------------------------
function string_drivers(drivers)
	s = ""
	for k,v in pairs(drivers) do
		s = s .. " " .. v 
	end
	s = s .. " "
	return s
end






-- method to change the type of the data, models etc 
-- TODO: move this to an "aux" file
function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end



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




-- Training function
---------------------------------------------

function train(trainer,excluded_drivers, epoch, print_stats)
	-- set up the confusion matrix
	confusion = optim.ConfusionMatrix(10)
	model = trainer.model 
	parameters = trainer.params 
	gradParameters = trainer.gParams 
	criterion = trainer.criterion 
	optimState = trainer.optimState 
	-- excluded_driver is an index, between 1 and 26 inclusive
	model:training()

	
	
	-- update on progress
	print(c.blue '==>'.." Traioning without:" .. string_drivers(excluded_drivers) .. " epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')


	-- get a set of batches of indices that don't include the excluded driver
	local valid_indices = torch.randperm(provider.data_n):long()

	for i, idx in pairs(excluded_drivers) do
		valid_indices = valid_indices[torch.ne(provider.driverIdx:index(1, valid_indices), idx)]
	end

	local perm_indices = torch.randperm (valid_indices:size(1)):long()
	local indices = valid_indices:index(1, perm_indices):long():split(opt.batchSize)

	local targets = cast(torch.FloatTensor(opt.batchSize))

	local tic = torch.tic()
	local total_loss = 0

	train_n = valid_indices:size(1)

	-- train on each batch
	for t,v in pairs(indices) do
		-- update progress
	    xlua.progress(t, #indices)
		if v:size(1) ~= opt.batchSize then
			break
		end
		-- set up batch

    		local inputs = provider.data:index(1,v)
			targets:copy(provider.labels:index(1,v))
			--local targets = provider.trainLabel:index(1,v)
			-- aluation function
	    	local feval = function(x)

				if x ~= parameters then parameters:copy(x) end      
      			gradParameters:zero()
      			local outputs = model:forward(inputs)
      			local f = criterion:forward(outputs, targets)
      			local df_do = criterion:backward(outputs, targets)
				total_loss = total_loss + f * opt.batchSize 
      			model:backward(inputs, df_do)
      			confusion:batchAdd(outputs, targets)
			

				--print("1for")			
				L2 = 0--torch.norm(parameters)
				L1 = 0--torch.sum(torch.abs(parameters))
				--print (f, L1, L2, f+opt.L1*L1+opt.L2*L2)
				f = f + opt.L2 * L2
				f = f + opt.L1 * L1
			
      			
      			-- return criterion output and gradient of the parameters
      			return f, gradParameters--(gradParameters + opt.L2 * 2 * parameters + opt.L1 * torch.sign(parameters))
    		end
	
		-- one iteration of the optimizer
		if opt.trainAlgo == 'sgd' then
    			optim.sgd(feval, parameters, optimState)
		elseif opt.trainAlgo == 'adam' then
			optim.adam(feval, parameters, optimState)	
		end
	end


	-- update confusion matrix
	confusion:updateValids()
	if print_stats then
		print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
    		confusion.totalValid * 100, torch.toc(tic)))
		print(('Train     loss: '..c.cyan'%.6f'):format(total_loss/train_n))
	end
	
	train_acc = confusion.totalValid * 100

	return train_acc, total_loss/train_n
	
end


-- Create submission file
-------------------------

function create_submission(model)

	print ("img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9")
	for i = 1,provider.test_n do	

		results = model:forward(provider.test[i])
		print ("")

	end	
		
end


-- Validate function
----------------------
function validate(model, excluded_drivers, print_stats, print_confmat)
	-- set up the confusion matrix
	confusion = optim.ConfusionMatrix(10)




	model:evaluate()
	print(c.blue '==>'.." validating on drivers " .. string_drivers(excluded_drivers) )
	local bs = opt.batchSize
	local total_loss = 0


	-- Get the set of "batches" where the drivers are excluded
	local valid_indices = torch.range(1,provider.data_n):long()
	local is_excluded = torch.zeros(provider.data_n):long()
	-- If an index points to an excluded driver, set the is_excluded to 1
	for i, idx in pairs(excluded_drivers) do
		is_excluded[torch.eq(provider.driverIdx:index(1, valid_indices), idx)] = 1
	end
	valid_indices = valid_indices[torch.eq(is_excluded, 1)]
	--print (valid_indices, is_excluded, provider.data_n)
	local perm_indices = torch.randperm (valid_indices:size(1)):long()
	local indices = valid_indices:index(1, perm_indices):long():split(opt.batchSize)

	for t,v in pairs(indices) do
		-- update progress
	    xlua.progress(t, #indices)


		-- set up batch
    	local inputs = provider.data:index(1,v)
		local targets = provider.labels:index(1,v)
		local outputs = model:forward(inputs)
		local loss = criterion:forward(outputs, targets)
		
		bs = inputs:size(1)
		-- fix for batchsize 1 
		if bs == 1 then
			outputs = outputs:reshape(1, outputs:size(1))		
			targets = targets:reshape(1, targets:size(1))
		end
	   	confusion:batchAdd(outputs, targets)
  		total_loss = total_loss + loss * bs
	end

	confusion:updateValids()
	if print_stats then
	  	print(('Valid accuracy: '..c.cyan'%.2f'):format(confusion.totalValid * 100))
  		print(('Valid loss: '..c.cyan'%.6f'):format(total_loss/valid_indices:size(1)))
	end
	if print_confmat then
	    print(confusion)
	end
	return confusion.totalValid, total_loss/valid_indices:size(1), valid_indices:size(1), confusion
end



---------------------------------


---------------------------------
--           Main              --  
---------------------------------


---------------------------------



-- Generate data file if needed
-----------------------------------------------
height = opt.height
width = opt.width
provider = 0
if opt.gen_data ~= "no" then 
	-- TODO move most of this to provider.lua
	num_train = -1
	provider = Provider("/home/tc/data/distracted-drivers/", num_train, height, width, false)
	provider:normalize()

	-- Setup bettertraining/testing sets
	-- Bone-head way: take fist n drivers for training, use last v for validation
	provider.labels = provider.labels+1

	collectgarbage()
	print (c.blue"Saving file...")
	torch.save(opt.datafile, provider)
end


-- Set up models/trainers
-------------------------
folds = torch.range(1,26):chunk(opt.n_folds)
-- TODO: Create a method for having multiple an overcomplete x-fold 
-- ie. drivers can appear in multiple folds, like in RF 
trainers = {}
print(c.blue '==>' ..' configuring model')
for i = 1,opt.n_folds do

	trainer = get_trainer()
	excluded_drivers = {}
	included_drivers = {}
	for j = 1,folds[i]:size(1) do
		table.insert(excluded_drivers, folds[i][j], folds[i][j])
	end
	for j = 1,26 do
		if excluded_drivers[j] == nil then
			table.insert(included_drivers, j, j)
		end
	end
	trainer.excluded_drivers = excluded_drivers
	trainer.included_drivers = included_drivers
	trainers[i] = trainer
end


-- Load the data
----------------------
print (c.blue"Loading data...")
provider = torch.load(opt.datafile)
provider.data = cast(provider.data)
provider.labels = cast(provider.labels)




-- Train / validate the model(s)
--------------------------------

for epoch = 1,opt.max_epoch do
	
	print (c.blue"Training epoch " .. epoch .. c.blue "  ---------------------------------")

	for fold = 1,opt.n_folds do
		print ("Training epoch " .. epoch .. " fold " .. fold)

		trainer = trainers[fold]
		-- train each model one epoch
		train(trainer, trainer.excluded_drivers, epoch, true)
	end


	print (c.blue"Validation epoch " .. epoch .. c.blue "  --------------------------------")
	--[[ Validation should print out:
		- Each model's accuracy / loss on its validation set
		- Aggregated validation set accuracy / loss
		- Aggregated class accuracy/loss
		- Aggregated driver acucracy/loss
		
		* Note, should account for when a class is excluded from multiple folds
	]]
	local aggregate = torch.Tensor(provider.data:size())
	aggregate[{}] = 1
	local total_loss = 0
	local total_acc = 0
	for fold = 1,opt.n_folds do
		print ("Validating epoch " .. epoch .. " fold " .. fold)
		acc, loss, n_valid = validate(trainers[fold].model, trainers[fold].excluded_drivers, false, false)
		-- TODO: use some nicer formatting
		print ("Fold " .. fold .. " (" .. string_drivers(trainers[fold].excluded_drivers).. ") \tacc = " .. acc*100 .. "\tloss = " .. loss)
		total_loss = total_loss + loss * n_valid
		total_acc = total_acc + acc * n_valid
	end
	print (c.Magenta"==>" .. " TOTAL \tacc = " ..  total_acc / provider.data_n * 100.0 .. "\t loss = " .. total_loss/provider.data_n)


	--print (c.blue"Logging epoch " .. epoch .. c.blue " ---------------")
	--[[ TODO Logging should consist of
		Each model's predictions on all data
		Saving every mode
		Confusion matrix, validation stats
	]]	

  	--save model every 10 epochs
	--[[
  	if epoch % 25 == 0 then
    	local filename = paths.concat(opt.save, 'model_' .. epoch .. '.net')
    	print('==> saving model to '..filename)
    	torch.save(filename, model:clearState())
  	end
	]]
	
	
	-- TODO: decrease learning rate over time
end




