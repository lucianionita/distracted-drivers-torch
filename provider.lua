require 'nn'
require 'image'


local c = require 'trepl.colorize'
local Provider = torch.class 'Provider'
torch.setdefaulttensortype('torch.FloatTensor')

function Provider:__init(folder, n_max, height, width, load_test_images)
	

	local img
	-- folder is where the training data is stored
	-- n_max is the maximum number of each class that is read from training
	-- also, 10*n_max is the maximum number of examples read from the test set

	
	-- list of data files, full path
	self.data_files = {}
	-- list of test files, full path
	self.test_files = {}
	-- Tensor of data files
	self.data = {}
	-- tensor of test files
	self.test = {}
	-- labels of data files
	self.labels = {}
	-- driver IDs of data files
	self.driverId = {}
	-- driver idx, ie numeric, driverId = drivers[driverIdx]
	self.driverIdx = {}


	-- easier to just list them here rather than acquire them automatically 
	self.drivers = {"p002", "p012", "p014", "p015", "p016", "p021", "p022", "p024"
		, "p026", "p035", "p039", "p041", "p042", "p045", "p047", "p049", "p050"
		, "p051", "p052", "p056", "p061", "p064", "p066", "p072", "p075", "p081"}
	-- drivers inverse table
	self.invDrivers = {}
	for k, d in pairs(self.drivers) do
		self.invDrivers[d] = k
	end

	-- read drivers file
	file = io.open(folder .. '/driver_imgs_list.csv','r')
	file:read()
	s = file:read()
	img_driver = {}
	repeat
		s = s:split(',')
		img_driver[s[3]] = s[1]
		s = file:read()
	until s == nil
	file:close()


	
	-- Get list of train file
	for class = 0, 9 do
		k = n_max
		for file in paths.files(folder .. '/train/c' .. class) do
			-- Avoid files like . and ..
			if file:find('jpg') then
				table.insert(self.data_files, paths.concat(folder .. '/train/c' .. class, file))
				table.insert(self.labels, class)
				table.insert(self.driverId, img_driver[file])
				table.insert(self.driverIdx, self.invDrivers[img_driver[file]])
				k = k - 1
				if k == 0 then
					break
				end
			end
		end
	end

	-- Convert the driverIdx to tensor
	self.driverIdx = torch.Tensor(self.driverIdx)
	

	-- Get the test image files
	k = n_max * 10
	for file in paths.files(folder .. '/test/') do		
		-- Avoid non-image files
		if file:find('jpg') then
			table.insert(self.test_files, paths.concat(folder .. '/test/', file))
			k = k - 1
			if k == 0 then
				break
			end
		end
	end	


	-- Sort test image files
	table.sort(self.test_files, function(a,b) return a<b end)


	-- Load training images 	
	local data = torch.FloatTensor(table.getn(self.data_files), 3, height, width)
	data:zero()
	print (c.blue"Loading training images")
	for i, file in ipairs(self.data_files) do
        	img = image.load(file)
        	img = image.scale(img, width, height)
        	img = img:reshape(1, 3, height, width)
		img = img:float()

		--table.insert(self.data, img)
		data[{{i},{},{},{}}] = img
		xlua.progress(i, table.getn(self.data_files))
	end
	
	self.data_n = data:size(1)
	self.data = data

	self.load_test_images = load_test_images

	if load_test_images then
		self:loadTestImages()
	end

	

	print (c.blue"Transforming tables to tensors")
	
	lbls = torch.Tensor(table.getn(self.labels))
	for i, l in ipairs(self.labels) do
		lbls[i] = l
	end
	self.labels = lbls
	print ("Done")
end


function Provider:loadTestImages()


		print (c.blue"Loading test images")

		local test = torch.Tensor(table.getn(self.test_files), 3, height, width)
		test:zero()
		for i, file in ipairs(self.test_files) do

			local img = image.load(file)
			img = image.scale(img, width, height)
   		 	img = img:reshape(1, 3, height, width)
					
			--table.insert(self.test, img)
			test[{{i},{},{},{}}] = img	
			xlua.progress(i, table.getn(self.test_files))
			if i % 500 == 0 then		
				collectgarbage()
			end
		end
	
		self.test = test
		self.test_n = test:size(1)

end

function Provider:normalizeTestImages()


  	print (c.blue"Normalizing test images")
  	local testData = self.test

  	local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
	-- preprocess testSet
	for i = 1,self.test_n do
		xlua.progress(i,self.test_n)
		-- rgb -> yuv		
		local rgb = testData[i]
		local yuv = image.rgb2yuv(rgb)
     		-- normalize y locally:
     		yuv[{1}] = normalization(yuv[{{1}}])
	  		-- normalize u globally:
  			yuv:select(1,2):add(-self.mean_u)
		  	yuv:select(1,2):div(self.std_u)
 		 	-- normalize v globally:
		  	yuv:select(1,3):add(-self.mean_v)
  			yuv:select(1,3):div(self.std_v)
     		testData[i] = yuv
  	end

  	self.test = testData

end


function Provider:normalize()
  -- TODO: normalize by taking the test images into consideration as well
  print (c.blue"Normalizing images")


  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  local trainData = self.data

  for i = 1,self.data_n do
     xlua.progress(i, self.data_n)
     -- rgb -> yuv
     local rgb = trainData[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData[i] = yuv
  end

  -- normalize u globally:
  local mean_u = trainData:select(2,2):mean()
  local std_u = trainData:select(2,2):std()
  trainData:select(2,2):add(-mean_u)
  trainData:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData:select(2,3):mean()
  local std_v = trainData:select(2,3):std()
  trainData:select(2,3):add(-mean_v)
  trainData:select(2,3):div(std_v)

  self.data = trainData
  
  self.mean_u = mean_u
  self.std_u = std_u
  self.mean_v = mean_v
  self.std_v = std_v

	if self.load_test_images then
		self:normalizeTestImages()
	end

end
