require 'xlua'
local iterUtil = {}

function iterUtil.initNormal()
	-- Initialize means and standard deviations
	local mu = torch.linspace(-4, 2, opt.mixture_size)
	local sigma = torch.Tensor(opt.mixture_size):fill(1)

	return mu, sigma
end

local function findWeights(loader)
	-- get next batch of inputs/targets
    local x, y = loader:next_batch()

    -- Initialize network
    local current_state = init_state

    -- Tensors to hold weight and target values
    local w = torch.Tensor(100, opt.batch_size, opt.mixture_size):zero()
    local target = torch.Tensor(100, opt.batch_size):zero()

    -- forward pass
    for t=1,120 do

        local lst = protos.rnn:forward{x[{{}, t}], unpack(current_state)}
        -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
        current_state = {}
        for i=1,state_size do table.insert(current_state, lst[i]) end

        if t > 20 then
        	local weights = torch.exp(lst[#lst]) -- find weights
        	
        	--store values
        	w[t - 20] = weights
        	target[t - 20] = y[{{}, t - 20}]
        end
    end
    return w, target
end

local function updateParams(ws, targets, total)

	-- Find new mus
	local y_repeat = torch.repeatTensor(targets, opt.mixture_size, 1):transpose(1, 2)
	local mu_num = torch.sum(torch.cmul(ws, y_repeat), 1)
	local mu_new = torch.cdiv(mu_num, torch.sum(ws, 1))

	-- Find new sigmas
	local diff = y_repeat - torch.repeatTensor(mu_new, total, 1)
	local num = torch.sum(torch.cmul(ws, torch.pow(diff, 2)), 1)
	local sigma_new = torch.sqrt(num/total)
	return mu_new, sigma_new
end

-- Function to update mean and std dev of distributions.
-- Can probably be done in larger batches to improve speed
function iterUtil.updateNormal(loader)

	-- Reset batch indices
	loader.batch_ix = {1, 0}
	loader.moreBatches = true

	-- variable to keep track of tensor size
	local total = 0

	-- Initialize tensors
	local ws = torch.Tensor(1e6, opt.mixture_size):zero()
	local targets = torch.Tensor(1e6):zero()
	local step = 100 * opt.batch_size

	-- Initialize new tensors
	local mu = torch.Tensor(opt.mixture_size)
	local sigma = torch.Tensor(opt.mixture_size)

	-- Initialize network
    init_state = {}
    for L=1,opt.num_layers do
        local h_init = torch.zeros(opt.batch_size, opt.nn_size)
        if opt.gpuid >=0 then h_init = h_init:cuda() end
        table.insert(init_state, h_init:clone())
        table.insert(init_state, h_init:clone())
    end
    state_size = #init_state

	-- Iterate over all batches in all folds in training set
	iterations = (opt.nfolds - 1) * loader.batches
	for i = 1, iterations do

		-- Find weights and target values, then reshape
		w, target = findWeights(loader)
		torch.reshape(w, step, opt.mixture_size)
		torch.reshape(target, step)

		-- Store values
		ws[{{total + 1, total + step}, {}}] = w
		targets[{{total + 1, total + step}}] = target
		total = total + step

		if i % 100 == 0 then 
			print(string.format('%.1f percent complete', i/iterations*100))
		end
	end

	-- Get rid of empty values
	ws = ws[{{1, total}, {}}]
	targets = targets[{{1, total}}]
	collectgarbage()

	mu_new, sigma_new = updateParams(ws, targets, total)

	print('new mu: ')
	print(mu_new)
	print('new sigma: ')
	print(sigma_new)

	-- Define new criterion with updated distribution
	protos.criterion = normalNLL(opt.mixture_size, mu_new, sigma_new)
end
















return iterUtil