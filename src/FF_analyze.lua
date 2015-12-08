require 'gnuplot'
require 'csvigo'
local FF_analyze = {}

-- Random sample from predicted distributions
local function sample(prediction, model)
    local bin_size = 8/opt.nbins
    local probs = torch.exp(prediction)
    local bins = torch.multinomial(probs, 1):double() -- sample bins from softmax
    return ((bins - 1) * bin_size):squeeze() + torch.rand(m) * bin_size - 5 -- sample within bin
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
            local prediction = model:forward(convert.augmentInput(x[{t, {}, {2, 5}}]))
            local acc = sample(prediction, model)

        	x[{t+1, {}, 1}] = x[{t, {}, 1}] + torch.mul(x[{t, {}, 4}], 0.1) + torch.mul(acc, 0.01) -- x_ego(t + dt)
            x[{t+1, {}, 2}] = -x[{t+1, {}, 1}] + x_lead[t - 20] -- d(t + dt)
            x[{t+1, {}, 4}] = x[{t, {}, 4}] + torch.mul(acc, 0.1) -- s(t + dt)
            x[{t+1, {}, 3}] = -x[{t+1, {}, 4}] + s_lead[t - 20] -- r(t + dt)
            x[{t+1, {}, 5}] = acc -- a(t + dt)
    end
    return x[{{22, 121}, {}, {2, 5}}]
end

-- Function to write tensors to csv file in desired directory
local function toFile(dir, data)
    csvigo.save('./' .. dir .. '/FF_softmax.csv', torch.totable(data))
end

-- Function to find the RWSE over different prediction horizons on the validation set
function FF_analyze.findError(loader)

    -- Load inputs/targets
    local states = loader.X[loader.valSet]
    local target = loader.vel[loader.valSet]
    local acc = loader.Y[loader.valSet]
    local x_lead = loader.x_lead[loader.valSet]
    local s_lead = loader.s_lead[loader.valSet]

    --Define # of trajectories to simulate
    m = 50

    states = torch.reshape(states, loader.batches*opt.batch_size, 120, 4)
    model:evaluate()

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
    toFile('analysis', torch.cat(sim, real, 2))
end

return FF_analyze