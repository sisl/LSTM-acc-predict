require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'normalNLL'
require 'csvigo'
convert = require '../util/convert'
local CarDataLoader = require '../util/CarDataLoader'

-- Load a network from checkpoint
function load_net(valSet)

	-- Specify directory containing stored nets
	local net_dir = '/Users/jeremymorton/Documents/AA_228/Final_Project/src/nets/ten_fold'

	-- Specify RNN checkpoint file
	checkpoint_file = net_dir .. '/FF_2_epochs' .. valSet .. '.00.t7'

	-- Load RNN from checkpoint
	print('loading a network from checkpoint')
	local checkpoint = torch.load(checkpoint_file)
	model = checkpoint.model
	model:evaluate()
	opt = checkpoint.opt

	print('Done.')
	collectgarbage()
end

-- Random sample from predicted distributions
local function sample(prediction, model)
    -- Single Gaussian component
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

-- Function to generate velocity predictions at 1, ..., 10 sec horizons
-- using recurrent neural network
local function propagate(states, target, x_lead, s_lead, loader)

    -- Create tensor to hold state at each time step, starting 2 seconds back
    -- state, x = [d(t), r(t), s(t), a(t)]
    local x = torch.zeros(121, m, 5)
    local input = torch.zeros(120, m, 4)

    -- Find lead vehicle position at t = 0
    local x_lead0 = x_lead[1] - 0.1*s_lead[1]

    -- Loop over all simulated trajectories
    for i = 1, m do
        -- Fill in vehicle position at last time step before propagation begins
       x[{21, i, 1}] = -states[{21, 1}] + x_lead0

       -- Fill in states at t = -2 sec to t = 0s
       x[{{1, 21}, i, {2, 5}}] = states[{{1, 21}, {}}]

       input[{{}, i, {}}] = states
    end

    -- forward pass
    for t=21,120 do
            local prediction = model:forward(convert.augmentFF(x[{{}, {}, {2, 5}}], t, loader))
            -- print(prediction)
            local acc = sample(prediction, model)
            -- acc = convert.alterAcc(acc, loader)

            x[{t+1, {}, 1}] = x[{t, {}, 1}] + torch.mul(x[{t, {}, 4}], 0.1) + torch.mul(acc, 0.01) -- x_ego(t + dt)
            x[{t+1, {}, 2}] = -x[{t+1, {}, 1}] + x_lead[t - 20] -- d(t + dt)
            x[{t+1, {}, 4}] = x[{t, {}, 4}] + torch.mul(acc, 0.1) -- s(t + dt)
            x[{t+1, {}, 3}] = -x[{t+1, {}, 4}] + s_lead[t - 20] -- r(t + dt)
            x[{t+1, {}, 5}] = acc -- a(t + dt)
    end
    return x[{{22, 121}, {}, {2, 5}}]
end

-- Function to write tensors to csv file in desired directory
local function toFile(dir, data, fold)
    csvigo.save('./' .. dir .. '/FF_mixture_' .. fold .. '_' .. opt.time_steps .. '.csv', torch.totable(data))
end

-- Load state, velcity, and acceleration data
loader = CarDataLoader.create(10, 10, true)

-- Loop over each fold in cross-validation
for fold = 4, 4 do

    -- Load inputs/targets
    loader.valSet = fold
    local states = loader.X[fold]
    local target = loader.vel[fold]
    local acc = loader.Y[fold]
    local x_lead = loader.x_lead[fold]
    local s_lead = loader.s_lead[fold]

    --Define # of trajectories to simulate
    m = 50

    -- Load network
    load_net(fold)
    
    -- Reshape data
    states = torch.reshape(states, loader.batches*opt.batch_size, 120, 4)

    -- True values of target variables to be simulated
    target = torch.reshape(target, loader.batches*opt.batch_size, 100) -- velocity
    acc = torch.reshape(acc, loader.batches*opt.batch_size, 100)
    x_lead = torch.reshape(x_lead, loader.batches*opt.batch_size, 100)
    s_lead = torch.reshape(s_lead, loader.batches*opt.batch_size, 100)
    collectgarbage()

    print('Propagating trajectories in fold ' .. fold)

    -- Initialize tensors to hold data
    local sim = torch.zeros(loader.batches*10, 100, 50, 4)
    local real = torch.zeros(loader.batches*10, 100, 4)
    local size = 0

    -- Loop over all inputs
    for i = 1, states:size(1) do
        if target[i][100] ~= 0 then -- Avoid wasting time on abbreviated trajectories
            size = size + 1

            -- Propagate simulated trajectories and store output
            sim[size] = propagate(states[i], target[i], x_lead[i], s_lead[i], loader)

            -- Combine and store true trajectory values
            local d = torch.cat(states[{i, {22, 120}, 1}], torch.Tensor({0}))
            local r = torch.cat(states[{i, {22, 120}, 2}], torch.Tensor({0}))
            local dr = torch.cat(d, r, 2)
            local va = torch.cat(target[i], acc[i], 2)
            real[size] = torch.cat(dr, va, 2)
        end
        if i%500 == 0 then print(i) end -- track progress
    end
    -- Get rid of empty values
    sim = sim[{{1, size}, {}, {}, {}}]
    real = real[{{1, size}, {}, {}}]

    -- Reshape tensors so that they can be written to csv
    sim = torch.reshape(sim, size * 100, m*4)
    real = torch.reshape(real, size * 100, 4)
    collectgarbage()

    -- Combine tensors, rescale and shift state values
    local combine = torch.cat(sim, real, 2)

    -- Write data to csv
    toFile('ten_fold', combine, fold)
end