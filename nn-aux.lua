

-- method to change the type of the data, models etc 
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



-- method to change the type of the data, models etc 
-- takes opt as argument 
function castopt(t, opt)
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



require 'image'

function distort(img, degrees, zoom_ratio, stretch_x, stretch_y, translate_x, translate_y)
	local height = img:size(2)
	local width = img:size(3)

	-- create canvas 	
	local canvas = torch.FloatTensor(img:size(1), height*2, width*2)
	canvas:zero()

	-- get scale
	local zoom_ratio = torch.uniform() * (zoom_ratio-1) + 1
	if (torch.uniform() < 0.5) then
		zoom_ratio = 1 / zoom_ratio
	end
	
	-- get stretch
	local stretch_x_ratio = torch.uniform() * (stretch_x-1) + 1
	if (torch.uniform() < 0.5) then
		stretch_x_ratio = 1 / stretch_x_ratio
	end
	local stretch_y_ratio = torch.uniform() * (stretch_y-1) + 1
	if (torch.uniform() < 0.5) then
		stretch_y_ratio = 1 / stretch_y_ratio
	end

	-- scale and stretch
	local target_width = math.floor(width * zoom_ratio * stretch_x_ratio)
	local target_height = math.floor(height * zoom_ratio * stretch_y_ratio)
	local simg = image.scale(img, target_width, target_height)

	-- paste on canvas
	local x = math.floor((width*2 - target_width)/2)
	local y = math.floor((height*2 - target_height)/2)
	canvas:narrow(2, y, target_height):narrow(3, x, target_width):copy(simg)

	-- perform rotation
	local angle_rad = (torch.uniform() * 2 - 1 ) * math.pi/180 * degrees
	canvas = image.rotate(canvas, angle_rad)

	-- crop the image out
	local dx = (torch.uniform() * 2 - 1 ) * translate_x
	local dy = (torch.uniform() * 2 - 1 ) * translate_y
	x = math.floor(width/2 + dx)
	y = math.floor(height/2 + dy)
	img_out = torch.FloatTensor(img:size(1),height, width)
    img_out:copy(canvas:narrow(2, y, height):narrow(3, x, width))

	return img_out
end
