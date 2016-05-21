-- Training function
---------------------------------------------

function train(trainer,excluded_drivers, epoch, fold, verbose, print_stats)
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



    -- update on progress
    if verbose then
		print(c.blue '==>'.." Training fold # " .. fold .. "/" .. opt.n_folds .. "\t sans (" .. string_drivers(excluded_drivers) .. ")\t epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	end

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
            -- TODO add distort here
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

                -- TODO: test this, it's experimental               
                L2 = torch.norm(parameters)  /  numParams
                L1 = torch.sum(torch.abs(parameters)) / numParams
                -- TODO: monitor L1,L2 norms print (f, L1, L2, f+opt.L1*L1+opt.L2*L2)

		if opt.batchStats then
	                print ("L1=" .. L1 .. "  L2=" .. L2 .. "  l=" .. f )	
		end
                f = f + opt.L2 * L2
                f = f + opt.L1 * L1

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
    end

    -- update confusion matrix
    confusion:updateValids()
    if print_stats then
        print(('Train accuracy: '..c.cyan'%.2f\tloss: '.. c.cyan'%.6f'.. '\t time: %.2f s'):format(
            confusion.totalValid * 100, torch.toc(tic),total_loss/train_n))
    end

    train_acc = confusion.totalValid

    return train_acc, total_loss/train_n, train_n

end


