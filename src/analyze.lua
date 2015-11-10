require 'gnuplot'
require 'csvigo'
local analyze = {}

-- Random sample from predicted distributions
local function sample(prediction, protos)
    if opt.mixture_size >= 1 then -- Gaussian mixture
        
        -- Iterative Gaussian mixture
        if opt.iter then 
            local n = prediction:size(2)
            local acc = torch.Tensor(m):zero()
            for i = 1, n do
                rand_samples = torch.randn(m) -- Random sample from unit normal dist
                pred = rand_samples * protos.criterion.sigma[i] + protos.criterion.mu[i] -- convert to normal w/ given mu and sigma
                acc = acc + torch.cmul(torch.exp(prediction[{{}, i}]), pred) -- apply weights
            end
            return acc

        -- Deep neural network prediction on Gaussian mixture
        else
            if prediction:size(2) == 2 then
                local rand_samples = torch.randn(m) -- generate m samples from standard normal distribution
                return torch.cmul(prediction[{{}, 2}], rand_samples) + prediction[{{}, 1}]
            else
                local n = prediction:size(2)/3
                local acc = torch.Tensor(m):zero()
                for i = 1, n do
                    rand_samples = torch.randn(m)
                    pred = torch.cmul(prediction[{{}, 2*n + i}], rand_samples) + prediction[{{}, n + i}] -- convert
                    acc = acc + torch.cmul(prediction[{{}, i}], pred)
                end
                return acc
            end
        end

    -- Softmax
    else
        local bin_size = 8/opt.nbins

        -- Iterative softmax
        if opt.iter then
            local w = torch.exp(prediction)
            local probs = torch.zeros(m, opt.nbins)
            for i = 1, m do
                local w_tensor = torch.repeatTensor(w[{i, {}}], opt.nbins, 1):transpose(1, 2)
                probs[{i, {}}] = torch.sum(torch.cmul(protos.criterion.pmf, w_tensor), 1):squeeze()
            end
            local bins = torch.multinomial(probs, 1):double() -- sample bins from softmax
            return ((bins - 1) * bin_size):squeeze() + torch.rand(m) * bin_size - 5 -- sample within bin

        -- Binned acceleration softmax 
        else
            local probs = torch.exp(prediction)
            local bins = torch.multinomial(probs, 1):double() -- sample bins from softmax
            return ((bins - 1) * bin_size):squeeze() + torch.rand(m) * bin_size - 5 -- sample within bin
        end
    end
end

-- Function to generate velocity predictions at 1, ..., 10 sec horizons
-- using recurrent neural network
local function propagate(states, target, x_lead, s_lead)

	-- Create tensor to hold state at each time step, starting 2 seconds back
	-- state, x = [d(t), r(t), s(t), a(t)]
	local x = torch.zeros(121, m, 5)
    local input = torch.zeros(120, m, 4)

    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        x = x:cuda()
        states = states:cuda()
        target = target:cuda()
        x_lead = x_lead:cuda()
        s_lead = s_lead:cuda()
    end

    -- Initialize internal state
    current_state = init_state

    -- Find lead vehicle position at t = 0
    local x_lead0 = x_lead[1] - 0.1*s_lead[1]

    -- Loop over all simulated trajectories
    for i = 1, m do
        -- Fill in vehicle position at last time step before propagation begins
       x[{21, i, 1}] = -states[{21, 1}] + x_lead0

       -- Fill in states at t = -2 sec to t = 0
	   x[{{1, 21}, i, {2, 5}}] = states[{{1, 21}, {}}]

       input[{{}, i, {}}] = states
    end

    -- forward pass
    for t=1,120 do
        local lst = protos.rnn:forward{input[{t, {}, {}}], unpack(current_state)}
        -- local lst = protos.rnn:forward{x[{t, {}, {2, 5}}], unpack(current_state)}
        -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
        current_state = {}
        for i=1,state_size do table.insert(current_state, lst[i]) end
        if t > 20 then -- Generate predictions after 2 sec
            local prediction = lst[#lst] -- last element holds the acceleration prediction
            local acc = sample(prediction, protos)

        	x[{t+1, {}, 1}] = x[{t, {}, 1}] + torch.mul(x[{t, {}, 4}], 0.1) + torch.mul(acc, 0.01) -- x_ego(t + dt)
            x[{t+1, {}, 2}] = -x[{t+1, {}, 1}] + x_lead[t - 20] -- d(t + dt)
            x[{t+1, {}, 4}] = x[{t, {}, 4}] + torch.mul(x[{t, {}, 5}], 0.1) -- s(t + dt)
            x[{t+1, {}, 3}] = -x[{t+1, {}, 4}] + s_lead[t - 20] -- r(t + dt)
            x[{t+1, {}, 5}] = acc -- a(t + dt)
        end   
    end
    return x[{{22, 121}, {}, {2, 5}}]
end

-- Function to write tensors to csv file in desired directory
local function toFile(dir, data)
    if opt.iter then
        if opt.mixture_size > 0 then
            csvigo.save('./' .. dir .. '/iter_mixture.csv', torch.totable(data))
            local params = torch.cat(protos.criterion.mu, protos.criterion.sigma, 2)
            csvigo.save('./' .. dir .. '/iter_mixture_params.csv', torch.totable(params))
        else
            csvigo.save('./' .. dir .. '/iter_softmax.csv', torch.totable(data))
            csvigo.save('./' .. dir .. '/iter_softmax_params.csv', torch.totable(protos.criterion.pmf))
        end
    else
        if opt.mixture_size > 0 then
            csvigo.save('./' .. dir .. '/mixture.csv', torch.totable(data))
        else
            csvigo.save('./' .. dir .. '/softmax.csv', torch.totable(data))
        end
    end
end


-- Function to store network output and true acceleration value throughout trajectory
local function storeOutput(states, target)

    -- Create inputs and outputs
    local out = torch.zeros(100, outputs + 1)
    local x = torch.zeros(120, 1, 4)
    x[{{}, 1, {}}] = states

    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        x = x:cuda()
        states = states:cuda()
        target = target:cuda()
    end

    -- Initialize network
    local current_state = {}
    for L=1,opt.num_layers do
        local h_init = torch.zeros(1, opt.nn_size)
        if opt.gpuid >=0 then h_init = h_init:cuda() end
        table.insert(current_state, h_init:clone())
        table.insert(current_state, h_init:clone())
    end
    local state_size = #init_state

    -- forward pass
    -- NOTE: input and output to the network differ by one time step i.e. if input corresponds to
    -- t = 21 then output will correspond to t = 22.  Mapping from t -> time: t = 21 <-> time = 0 sec
    for t=1,120 do
        local lst = protos.rnn:forward{x[{t, {}, {}}], unpack(current_state)}
        current_state = {}
        for i=1,state_size do table.insert(current_state, lst[i]) end

        if t > 20 then -- Store values after 2 sec
            if opt.mixture_size > 0 and not opt.iter then
                out[{t - 20, {1, outputs}}] = lst[#lst] -- store network output
            else
                out[{t - 20, {1, outputs}}] = torch.exp(lst[#lst]) -- exponential on softmax
            end
            out[{t - 20, outputs + 1}] = target[t - 20] -- store true acceleration value
        end   
    end

    -- Store outputs and distribution parameters
    toFile('outputs', out)
    
end

-- Function to find the RWSE over different prediction horizons on the validation set
function analyze.findError(loader)

    -- Load inputs/targets
    local states = loader.X[loader.valSet]
    local target = loader.vel[loader.valSet]
    local acc = loader.Y[loader.valSet]
    local x_lead = loader.x_lead[loader.valSet]
    local s_lead = loader.s_lead[loader.valSet]

    --Define # of trajectories to simulate
    m = 5

    -- Initialize network
    init_state = {}
    for L=1,opt.num_layers do
        local h_init = torch.zeros(m, opt.nn_size)
        if opt.gpuid >=0 then h_init = h_init:cuda() end
        table.insert(init_state, h_init:clone())
        table.insert(init_state, h_init:clone())
    end
    state_size = #init_state
    states = torch.reshape(states, loader.batches*opt.batch_size, 120, 4)
    protos.rnn:evaluate()

    -- True values of target variables to be simulated
    target = torch.reshape(target, loader.batches*opt.batch_size, 100) -- velocity
    acc = torch.reshape(acc, loader.batches*opt.batch_size, 100)
    x_lead = torch.reshape(x_lead, loader.batches*opt.batch_size, 100)
    s_lead = torch.reshape(s_lead, loader.batches*opt.batch_size, 100)
    collectgarbage()

    -- Create for writing out generated data
    local sim = torch.zeros(states:size(1), 100, m, 4)
    local real = torch.zeros(states:size(1), 100, 4)
    local size = 0

    -- Loop over all inputs
    for i = 1, states:size(1) do
        if target[i][100] ~= 0 then -- Avoid wasting time on abbreviated trajectories
            size = size + 1
            if size == 1 then
                -- storeOutput(states[i], acc[i])
            end

            -- Propagate simulated trajectories and store output
            sim[size] = propagate(states[i], target[i], x_lead[i], s_lead[i])

            -- Store true trajectory values
            local d = torch.cat(states[{i, {22, 120}, 1}], torch.Tensor({0}))
            local r = torch.cat(states[{i, {22, 120}, 2}], torch.Tensor({0}))
            local dr = torch.cat(d, r, 2)
            local va = torch.cat(target[i], acc[i], 2)
            real[size] = torch.cat(dr, va, 2)
        end
        if i%500 == 0 then print(i) end
    end

    -- Get rid of empty values
    sim = sim[{{1, size}, {}, {}, {}}]
    real = real[{{1, size}, {}, {}}]

    -- Reshape tensors so that they can be written to csv
    sim = torch.reshape(sim, size * 100, m*4)
    real = torch.reshape(real, size * 100, 4)
    collectgarbage()

    -- Write data to csv
    toFile('analysis/true_state/', torch.cat(sim, real, 2))
end

return analyze