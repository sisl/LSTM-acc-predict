local convert = {}

-- Perform search along bin boundaries to find correct bin for acceleration value
local function findBin(val)
	local lo = 1
	local hi = bounds:size(1)

	-- Check for values on the boundary
	if val < bounds[lo] then
		return lo
	elseif val > bounds[hi] then
		return hi - 1
	else
		-- Continuously cut search space in half until bin is found
		while true do
			local med = math.floor((lo + hi)/2)
			if med == lo then
				return lo
			end

			-- Revise bounds
			if bounds[med] < val then
				lo = med
			else
				hi = med
			end
		end
	end
end

-- Convert target output values to indicators
function convert.toBins(y, nbins)
	local y_bytes = torch.zeros(#y):byte()
	for i = 1, y_bytes:size(1) do
		y_bytes[i] = findBin(y[i])
	end
	return y_bytes
end

-- Shift and scale inputs to the network so that they are zero-mean and normalized
local function shift_scale(x, loader)
	if x:size(2) == 4 then
		x = x - torch.repeatTensor(loader.shift, x:size(1))
		return torch.cdiv(x, torch.repeatTensor(loader.scale, x:size(1)))
	else
		x = x - torch.repeatTensor(loader.shift, x:size(1), x:size(2)/4)
		return torch.cdiv(x, torch.repeatTensor(loader.scale, x:size(1), x:size(2)/4))
	end
end

-- Find location of bin boundaries in acceleration distribution using equal frequency binning
function convert.findBoundaries(loader, nbins)
	-- local accDiff = findDiff(loader) -- difference between successive values
	acc = loader.Y:view(loader.Y:size(1) * loader.Y:size(2) * loader.Y:size(3) * loader.Y:size(4))
	acc = torch.sort(acc) -- Sort values in ascending order

	-- Remove values exactly equal to zero
	acc = acc[acc:ne(0)]

	-- Find indices for boundaries
	indxs = torch.round(torch.linspace(1, acc:size(1), nbins + 1))

	-- Create mask to extract entries and apply mask
	mask = torch.ByteTensor(#acc)
	for i = 1, indxs:size(1) do
		mask[indxs[i]] = 1
	end

	-- Apply mask and truncate extreme values (move outer boundaries inward)
	local bounds = acc[mask]
	bounds[1] = -5
	bounds[nbins+1] = 3

	-- Find compromise between equal width and equal frequency binning
	bounds = bounds * 0.5 + torch.linspace(-5, 3, nbins + 1) * 0.5
	return bounds
end

-- Return shifted and scaled inputs
function convert.augmentInput(x, loader)
	-- return shift_scale(x, loader)
	x = shift_scale(x, loader)
	return x[{{}, {4}}]
end

-- Augment input to FFNN to contain inputs from multiple time steps if desired
function convert.augmentFF(x, t, loader)
	if x:size(1) < x:size(2) then
		-- Initialize new input
		local new_input = torch.zeros(x:size(1), 4*opt.time_steps)
		for i = 1, opt.time_steps do
			new_input[{{}, {4*i - 3, 4*i}}] = x[{{}, t - i + 1}]
		end
		return shift_scale(new_input, loader)
	else
		-- Initialize new input
		local new_input = torch.zeros(x:size(2), 4*opt.time_steps)
		for i = 1, opt.time_steps do
			new_input[{{}, {4*i - 3, 4*i}}] = x[t - i + 1]
		end
		return shift_scale(new_input, loader)
	end
end

return convert