require 'cudnn'

Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local status, tds = pcall(require, 'tds')
tds = status and tds or nil


local DataLoader = torch.class 'DataLoader'

function DataLoader:__init(opt, provider)

	-- configure something regarding opt here 
	local function init()
		pcall(require, 'tds')
		require 'provider'
		require 'dataloader'
		require 'nn-aux'
	--print ("Loading data...")
	--provider2 = torch.load(opt.datafile)
	--provider2.data = castopt(provider2.data, opt)
	--provider2.labels = castopt(provider2.labels, opt)

    	--require('datasets/' .. opt.dataset)
   	end

   	local function main(idx)
      	torch.setnumthreads(1)
		_G.provider = provider
   	end
	
	self.threads = Threads(opt.nThreads, init, main)
	
end


function DataLoader:getNMinibatches()
	return self.nMinibatches
end

function DataLoader:getMinibatches(excluded_drivers)


    -- get a set of batches of indices that don't include the excluded driver
    local valid_indices = torch.randperm(provider.data_n):long()

    for i, idx in pairs(excluded_drivers) do
        valid_indices = valid_indices[torch.ne(provider.driverIdx:index(1, valid_indices), idx)]
    end

    local perm_indices = torch.randperm (valid_indices:size(1)):long()
    local indices = valid_indices:index(1, perm_indices):long():split(opt.batchSize)


    local tic = torch.tic()

    train_n = valid_indices:size(1)
	self.nMinibatches = #indices

	
	local size = provider.data_n
	local idx = 0 
	local k = 0
	self.test = "test";
	function enqueue()		
		while idx + opt.batchSize-1 <= size and self.threads:acceptsjob() and indices[k+1]~=nil do
			k = k + 1
			if (indices[k]:size(1) == opt.batchSize) then
				self.threads:addjob(
					function(startIdx,  v, opt)
						provider = _G.provider
						minibatch = provider.data:index(1,v)
						
						if (opt.distort) then
							for i = 1,opt.batchSize do
								minibatch[i] = distort(minibatch[i], opt.dt_angle, opt.dt_scale, 
											opt.dt_stretch_x, opt.dt_stretch_y, opt.dt_trans_x, opt.dt_trans_y)
							end
						end

    					targets = castopt(torch.FloatTensor(opt.batchSize), opt)
						targets:copy(provider.labels:index(1,v))
						return minibatch, targets
					end,
					function ( _minibatch_, _labels_ )
						self.next_minibatch = _minibatch_
						self.next_labels = _labels_
					end,
					idx,
					indices[k], 
					opt
				)
			end
			idx = idx + opt.batchSize
		end
	end



	local n = 0 
	local function loop()
		enqueue()
		if not self.threads:hasjob() then
			return nil
		end
		self.threads:dojob()
		if self.threads:haserror() then
			self.threads:synchronize()
		end
		enqueue()
		n = n + 1 
		return n, {data = self.next_minibatch, labels = self.next_labels}
	end

	return loop

end














