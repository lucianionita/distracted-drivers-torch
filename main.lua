-- To create a new data file, this is what you need to do:
-- TODO Verify submission method
-- TODO Implement overcomplete crossval
-- TODO Implement conf mat for ensemble model for validation set
-- TODO Save models
-- TODO Save logs
-- TODO Try other training algos

-- Uncomment this is running with qlua
--require 'trepl'
--arg = {}
--arg[1] = '--gen_data'
require 'provider'
require 'xlua'
require 'optim'
require 'nn'
require 'dataloader'
c = require 'trepl.colorize'
torch.setdefaulttensortype('torch.FloatTensor')

-- Pull in requirements
-------------------------
require ('nn-aux')
require ('dd-aux')
require ('trainer')
require ('submission')
require ('validate')



-- Parse the command line arguments
--------------------------------------
opt = lapp[[
	--model 				(default linear_logsoftmax) 	model name
	-b,--batchSize 			(default 32) 			batch size
 	-r,--learningRate 		(default 1) 		learning rate
 	--learningRateDecay 	(default 1e-7) 		learning rate decay
	--lr_schedule			(default 100)		learning rate reduction schedule, how many epochs between LR decreases
	--lr_factor				(default 0.5)		learning rate reduction factor
	
	-s,--save 	(default "logs") 		subdirectory to save logs
	-S,--submission						generate(overwrites) submission.csv file

	-f,--n_folds	(default 3)				number of folds to use
	-g,--gen_data 				 			whether to generate data file 
	-d,--datafile 	(default p.t7) 			file name of the data provider
	-h,--height	(default 48)				height of the input images
	-w,--width	(default 64)				width of the resized images
	--L2		(default 0)					L2 norm
	--L1		(default 0)					L1 norm
	--num_train		(default -1)				Artificially reduces training set (DEBUG)

	-t,--trainAlgo	(default sgd)			training algorithm: sgd, adam
	--weightDecay 	(default 0.0005) 		weightDecay (use in SGD instead of L2)
	-m,--momentum 	(default 0.9) 			momentum
	--max_epoch 	(default 300) 			maximum number of iterations
	
	--nThreads		(default 2)				number of loader threads
	--distort								use distortions
	--dt_angle		(default 10)			distortion: angle of rotation in degrees
	--dt_scale		(default 1.1)			distortion: max scale factor (zoom in/out)
	--dt_stretch_x	(default 1.2)			distortion: max X stretch (ratio)
	--dt_stretch_y	(default 1.2)			distortion: max Y stretch (ratio)
	--dt_trans_x	(default 4)				distortion: max X translation (pixels)	
	--dt_trans_y	(default 4)				distortion:	max Y translation (pixels)

	--batchStats 							Stats after each batch	

 	--backend (default cudnn) 			backend to be used nn/cudnn
 	--type (default cuda) 				cuda/float/cl
	
	-v,--validation (default 6) 			number of drivers to use in validation set
]]


if opt.backend == 'cudnn' then
	   require 'cudnn'
end

if opt.distort then
	-- TODO currently storing the data is inefficient
	-- Provider's data should be on the CPU side and only sending it to the GPU after distortion
	require ('train') 
else
	-- if not using distortion, this is a faster at loading the data 
	require ('train_old')
end



---------------------------------


---------------------------------
--           Main              --  
---------------------------------


---------------------------------




-- Generate data (and save it to file)
-----------------------------------------------
height = opt.height
width = opt.width
provider = 0

-- If generating data
if opt.gen_data then
	load_test_set = false 
	if opt.submission then
		load_test_set = true
	end
	num_train = -1
	provider = Provider("/home/tc/data/distracted-drivers/", opt.num_train, 
				height, width, load_test_set)
	provider:normalize()
	
	-- Because lua is ONE-indexed
	provider.labels = provider.labels+1

	collectgarbage()
	xprint (c.blue"Saving file...")
	torch.save(opt.datafile, provider)
end


-- Load the data
----------------------
print (c.blue"Loading data...")
provider = torch.load(opt.datafile)
if not opt.distort then
	provider.data = cast(provider.data)
end
provider.labels = cast(provider.labels)

-- Create the dataloader
------------------------
dataloader = DataLoader(opt, provider)


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






-- Train / validate the model(s)
--------------------------------

for epoch = 1,opt.max_epoch do
	
	print (c.blue"=====>" .. " Epoch " .. epoch .. c.blue " <===================================================================================================")

	
	local total_train_acc = 0
	local total_valid_acc = 0
	local total_train_loss = 0 
	local total_valid_loss = 0
	local acc, loss
	local train_n = 0

	for fold = 1,opt.n_folds do

		trainer = trainers[fold]

		print(c.blue '==>'.." Training/validating on fold # " .. fold .. "/" .. opt.n_folds .. "\t (" .. string_drivers(trainer.excluded_drivers) .. ")\t epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

		-- train each model one epoch
		train_acc, train_loss, n = train(trainer, trainer.excluded_drivers, epoch, fold, true, false, true)
		total_train_acc = total_train_acc + train_acc * n
		total_train_loss = total_train_loss + train_loss * n 
		train_n = train_n + n
		
		-- validate one epoch		
		acc, loss, n = validate(trainers[fold].model, trainers[fold].excluded_drivers, false, false, true)
		total_valid_acc = total_valid_acc + acc * n
		total_valid_loss = total_valid_loss + loss * n
        
		print(('Train accuracy: '..c.cyan'%.2f' .. '\tloss: '.. c.cyan'%.6f'):format(train_acc * 100, train_loss))
        print(('Valid accuracy: '..c.green'%.2f' .. '\tloss: '.. c.green'%.6f' ):format(acc * 100, loss))
	
	end

    print((c.Magenta"==>" .. ' Epoch '.. epoch ..' Train accuracy: '..c.Magenta'%.2f%%' .. '\tloss: '.. c.Magenta'%.6f'.. ''):format(total_train_acc * 100 / (1.0*train_n), total_train_loss/train_n))
    print((c.Magenta"==>" .. ' Epoch '.. epoch ..' Valid accuracy: '..c.Magenta'%.2f%%' .. '\tloss: '.. c.Magenta'%.6f'.. ''):format(total_valid_acc * 100 / (1.0*provider.data_n), total_valid_loss/provider.data_n))

	--[[ TODO Validation should print out:
		- Each model's accuracy / loss on its validation set
		- Aggregated validation set accuracy / loss
		- Aggregated class accuracy/loss
		- Aggregated driver acucracy/loss
		
		* Note, should account for when a class is excluded from multiple folds
	]]

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
	
	-- Update trainers
	------------------------
	
	-- learning rate decay
	if opt.lr_schedule > 0 then
		if epoch % opt.lr_schedule == 0 then
			print (c.blue'==>' .. 'Reducing learning rate. Decay is ' .. opt.lr_factor)
			for i, trainer in pairs(trainers) do
				trainer.optimState.learningRate = trainer.optimState.learningRate * opt.lr_factor
			end			
		end
	end

	
end






-- Create submission file
--------------------------

if opt.submission then
	predictions = create_submission(trainers)
end
