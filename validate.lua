-- Validate function
----------------------
function validate(model, excluded_drivers, verbose, print_stats, print_confmat)
    -- set up the confusion matrix
    confusion = optim.ConfusionMatrix(10)




    model:evaluate()
	if verbose then
	    print(c.blue '==>'.." Validating on drivers (" .. string_drivers(excluded_drivers)..")" )
	end
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

