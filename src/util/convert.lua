local convert = {}

-- Convert target output values to indicators
function convert.toBins(y, nbins)
	local bin_size = 8/(nbins)
	local transform = torch.floor(torch.div(y + 5, bin_size)) + 1 -- Convert to bin number value

	-- Create byte tensor indicating bin value and fill in
	-- I think you can only fill in through iterating through y
	y_bytes = torch.ByteTensor(#y)
	for i = 1, y:size(1) do
		if transform[i] > nbins then transform[i] = nbins end
		if transform[i] <= 0 then transform[i] = 1 end 
		y_bytes[i] = transform[i]
	end
	return y_bytes
end

-- Acceleration prediction based on intelligent driver model
local function IDMpredict(x)

	-- Define parameters
	local n = x:size(2)
	local dmn = 5.301
	local T = 1.023
	local bcmf = 3.621
	local smx = 17.281
	local amx = 3

	-- Extract state values
	d = torch.cmax(x[{{}, 1}], 0.01)
	r = x[{{}, 2}]
	s = x[{{}, 3}]

	-- Find IDM acceleration predictions
	local d_des = s*T - torch.cmul(s, r)/2/math.sqrt(amx*bcmf) + dmn
	local pred = (-torch.pow(s/smx, 4) - torch.cmin(torch.pow(torch.cdiv(d_des, d), 2), 100) + 1) * amx
	pred = torch.cmax(pred, -5)
	return torch.cmin(pred, 3)
end

-- Augment input to network.  Currently constructed to add IDM acceleration prediction
function convert.augmentInput(x)
	-- Initialize new input
	local new_input = torch.zeros(x:size(1), x:size(2) + 1)
	new_input[{{}, {1, x:size(2)}}] = x
	new_input[{{}, x:size(2) + 1}] = IDMpredict(x)
	return new_input
end

-- Augment input to FFNN to contain inputs from multiple time steps if desired
function convert.augmentFF(x, t)
	if x:size(1) < x:size(2) then
		-- Initialize new input
		local new_input = torch.zeros(x:size(1), 20)
		new_input[{{}, {1, 4}}] = x[{{}, t - 20}]
		new_input[{{}, {5, 8}}] = x[{{}, t - 16}]
		new_input[{{}, {9, 12}}] = x[{{}, t - 12}]
		new_input[{{}, {13, 16}}] = x[{{}, t - 8}]
		new_input[{{}, {17, 20}}] = x[{{}, t - 4}]
		return new_input
	else
		-- Initialize new input
		local new_input = torch.zeros(x:size(2), 20)
		new_input[{{}, {1, 4}}] = x[t - 20]
		new_input[{{}, {5, 8}}] = x[t - 16]
		new_input[{{}, {9, 12}}] = x[t - 12]
		new_input[{{}, {13, 16}}] = x[t - 8]
		new_input[{{}, {17, 20}}] = x[t - 4]
		return new_input
	end
end

return convert