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


function train2() 

	best_model = nil;
	best_loss = 100000;

	trainer = get_trainer()
	trainer.excluded_drivers = {}
	trainer.included_drivers = {}
	for idx = 1,26 do
		isin_excluded = false
		for k, vv in pairs(opt.exclude:split(',')) do
			v = tonumber(vv)
			if v == idx then
				isin_excluded = true
			end
		end
		if isin_excluded then	
			table.insert(trainer.excluded_drivers, idx)
		else
			table.insert(trainer.included_drivers, idx)
		end
	end		
	
	for epoch = 1, opt.max_epoch do
	
	
		print(c.blue '==>'.." Training/validating epoch # " .. epoch .. "/" .. opt.max_epoch .. "\t (" .. string_drivers(trainer.included_drivers) .. ' [batchSize = ' .. opt.batchSize .. ']')
	
		-- train each model one epoch
		train_acc, train_loss, n = train(trainer, trainer.excluded_drivers, epoch, 0, true, false, true)
	
		print(('Train accuracy: '..c.cyan'%.2f' .. '\tloss: '.. c.cyan'%.6f'):format(train_acc * 100, train_loss))
		
		if table.getn(trainer.excluded_drivers) ~= 0 then	
			acc, loss, n = validate(trainer.model, trainer.excluded_drivers, false, false, true)
		    print(('Valid accuracy: '..c.green'%.2f' .. '\tloss: '.. c.green'%.6f' ):format(acc * 100, loss))
		end
		
		if opt.lr_schedule > 0 then
			if epoch % opt.lr_schedule == 0 then
				print (c.blue'==>' .. 'Reducing learning rate. Decay is ' .. opt.lr_factor)
				trainer.optimState.learningRate = trainer.optimState.learningRate * opt.lr_factor
			end
		end

		if (loss < best_loss) then
			print ("Got better loss: " ..  tostring(loss))
			best_loss = loss
			print ("Saving to: " .. opt.save_model)
			torch.save(opt.save_model, model)
		end
	end

end	
