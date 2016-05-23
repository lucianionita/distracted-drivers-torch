-- Training function
---------------------------------------------

function train(trainer,excluded_drivers, epoch, fold, verbose, print_stats, print_confmat)
	-- set up the confusion matrix
	confusion = optim.ConfusionMatrix(10)
	model = trainer.model
	parameters = trainer.params
	gradParameters = trainer.gParams
	criterion = trainer.criterion
	optimState = trainer.optimState
	numParams = parameters:size(1)
	-- excluded_driver is an index, between 1 and 26 inclusive
	model:training()

	local tic = torch.tic()


	-- update on progress
	if verbose then
		print(c.blue '==>'.." Training fold # " .. fold .. "/" .. opt.n_folds .. "\t sans (" .. string_drivers(excluded_drivers) .. ")\t epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	end

	local total_loss = 0
	for t, mb in dataloader:getMinibatches(excluded_drivers) do

		-- update progress
		xlua.progress(t, dataloader:getNMinibatches())

		-- set up batch

		local inputs = mb.data
		local targets = mb.labels

		local L1 = 0
		local L2 = 0
		local L1g = 0
		local L2g = 0
		local loss = -1

		-- Evaluation function
		local feval = function(x)

				if x ~= parameters then parameters:copy(x) end
				gradParameters:zero()
				local outputs = model:forward(inputs)
				local f = criterion:forward(outputs, targets)
				local df_do = criterion:backward(outputs, targets)
				total_loss = total_loss + f * opt.batchSize
				model:backward(inputs, df_do)
				confusion:batchAdd(outputs, targets)




				-- update stats
				if loss == -1 then
					loss = f
					L2g = torch.norm(gradParameters) / numParams
					L1g = torch.sum(torch.abs(gradParameters)) / numParams
				end
				loss = loss * 0.98 + f * 0.02
				L2g = L2g * 0.98 + 0.02 * torch.norm(gradParameters) / numParams
				L1g = L1g * 0.98 + 0.02 * torch.sum(torch.abs(gradParameters)) / numParams

				-- TODO: test this, it's experimental		       
				L2 = torch.norm(parameters)  /  numParams
				L1 = torch.sum(torch.abs(parameters)) / numParams

				if opt.batchStats then
							print ("L1=" .. L1 .. "  L2=" .. L2 .. "  l=" .. f )	
				end
		   
				-- add L1/L2 to f 
				f = f + opt.L2 * L2
				f = f + opt.L1 * L1

				-- add L1 and L2 derivs
				gradParameters:add(opt.L2 * 2, parameters)
				gradParameters:add(opt.L1, torch.sign(parameters))

				-- return criterion output and gradient of the parameters
				return f, gradParameters

			end

		-- one iteration of the optimizer
		if opt.trainAlgo == 'sgd' then
			optim.sgd(feval, parameters, optimState)
		elseif opt.trainAlgo == 'adam' then
			optim.adam(feval, parameters, optimState)
		end

		-- print stats TODO add a command line optino for this
		if verbose then
			if (t % 100) == 0 then
				print ("\r		                                                                                                                       \rMinibatch " .. t.. c.Blue"\tL1="  .. L1 .. c.Blue"\tL2=" ..  L2 .. c.Blue"\tl=" .. loss .. c.Blue"\tL1g=" .. L1g .. c.Blue"\tL2g=" .. L2g)
			end
		end

	end

	-- update confusion matrix
	confusion:updateValids()
	if print_stats then
		print(('Train accuracy: '..c.cyan'%.2f\tloss: '.. c.cyan'%.6f'.. '\t time: %.2f s'):format(
			confusion.totalValid * 100, torch.toc(tic),total_loss/train_n))
	end

	if print_confmat then
		print(confusion)
	end


	train_acc = confusion.totalValid

	return train_acc, total_loss/train_n, train_n

end


