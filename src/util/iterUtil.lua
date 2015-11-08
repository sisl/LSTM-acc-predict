local convert = require 'util.convert'
local iterUtil = {}

function iterUtil.initNormal()
	-- Initialize means and standard deviations
	local mu = torch.linspace(-4, 2, opt.mixture_size)
	local sigma = torch.Tensor(opt.mixture_size):fill(1)

	return mu, sigma
end

function iterUtil.initMultinomial()
	local n = opt.multinomial_size
	local probs = torch.zeros(n, opt.nbins)
	local resid = opt.nbins % n
	local bins = math.floor(opt.nbins/n) -- number of bins for each section of distribution
	local indices = torch.ones(n) -- Keep track of indices for each distribution
	for i = 1, n do
		for j = 1, n do
			if i == j then
				probs[{j, {indices[j], indices[j] + bins + resid - 1}}] = (1 - 1/n)/(bins + resid)
				indices[j] = indices[j] + bins + resid
			else
				probs[{j, {indices[j], indices[j] + bins - 1}}] = 1/(n^2 - n)/bins
				indices[j] = indices[j] + bins
			end
		end
	end
	return probs
end

local function findWeights(input)

    -- Tensors to hold weight and target values
    local w = torch.zeros(input:size(1), 100, outputs)

    -- Initialize internal state of network
    current_state = init_state

    -- forward pass
    for t=1,120 do

        local lst = protos.rnn:forward{input[{{}, t}], unpack(current_state)}
        current_state = {}
        for i=1,state_size do table.insert(current_state, lst[i]) end

        if t > 20 then
        	local weights = torch.exp(lst[#lst]) -- find weights
        	
        	--store values
        	w[{{}, t - 20}] = weights

        	if (t - 20) % 25 == 0 then 
        		print(string.format("%d percent done", t-20))
        	end
        end
    end
    return w
end

-- Update parameters in Gaussian mixture
local function updateParams(ws, targets)

	-- Find new mus
	local y_repeat = torch.repeatTensor(targets, opt.mixture_size, 1):transpose(1, 2)
	local mu_num = torch.sum(torch.cmul(ws, y_repeat), 1)
	local mu_new = torch.cdiv(mu_num, torch.sum(ws, 1))

	-- Find new sigmas
	local diff = y_repeat - torch.repeatTensor(mu_new[{1, {}}], targets:size(1), 1)
	local num = torch.sum(torch.cmul(ws, torch.pow(diff, 2)), 1)
	local sigma_new = torch.sqrt(num/targets:size(1))
	return mu_new[{1, {}}], sigma_new[{1, {}}]
end

-- Update probabilities in multinomial distribution
local function updateProbs(ws, targets)
	local new_probs = torch.zeros(ws:size(2), opt.nbins)
	for i = 1, ws:size(1) do
		new_probs[{{}, targets[i]}] = new_probs[{{}, targets[i]}] + ws[i]
	end
	local sums = torch.sum(new_probs, 2)
	return torch.cdiv(new_probs, torch.repeatTensor(sums[{{}, 1}], opt.nbins, 1):transpose(1, 2))
end

-- Function to take all inputs outside validation set and put them in one tensor
local function getXY(loader)
	local step = loader.X:size(2) * loader.X:size(3)
	local input = torch.zeros((loader.nfolds - 1) * step, 120, 4)
	local targets = torch.zeros((loader.nfolds - 1) * step, 100)
	local count = 0
	for i = 1, loader.nfolds do
		if i ~= loader.valSet then
			count = count + 1
			local x = torch.reshape(loader.X[i], step, 120, 4)
			local start = (count - 1) * step + 1
			input[{{start, count * step}, {}, {}}] = x

			local y = torch.reshape(loader.Y[i], step, 100)
			targets[{{start, count * step}, {}}] = y
		end
	end
	return input, targets
end

-- Function to update mean and std dev of distributions.
-- Can probably be done in larger batches to improve speed
function iterUtil.updateDist(loader)

	-- Get input tensor
	local dim = (loader.nfolds - 1) * loader.X:size(2) * loader.X:size(3)
	local input, targets = getXY(loader)

	-- Reshape and convert target tensor (if necessary)
	targets = torch.reshape(targets, targets:size(1)*100)
	if opt.nbins > 0 then
        targets = convert.toBins(targets, opt.nbins)
    end
    collectgarbage()

	-- Initialize network
    init_state = {}
    for L=1,opt.num_layers do
        local h_init = torch.zeros(dim, opt.nn_size)
        if opt.gpuid >=0 then h_init = h_init:cuda() end
        table.insert(init_state, h_init:clone())
        table.insert(init_state, h_init:clone())
    end
    state_size = #init_state

	-- Find weights and target values, then reshape
	ws = findWeights(input)
	ws = torch.reshape(ws, dim * 100, opt.mixture_size + opt.multinomial_size)
	collectgarbage()

	-- Update multinomial distribution
	if opt.nbins > 0 then
		probs_new = updateProbs(ws, targets)
		print('new probs: ')
		print(probs_new)
		-- Define new criterion with updated distribution
		protos.criterion = iterativeNLL(opt.nbins, probs_new)

	-- Update Gaussian mixture parameters
	else
		mu_new, sigma_new = updateParams(ws, targets)
		print('new mu: ')
		print(mu_new)
		print('new sigma: ')
		print(sigma_new)
		-- Define new criterion with updated distribution
		protos.criterion = normalNLL(opt.mixture_size, mu_new, sigma_new)
	end
end


return iterUtil