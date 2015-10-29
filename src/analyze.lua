require 'gnuplot'
local analyze = {}

-- Random sample from predicted distributions
local function sample(prediction)
    if opt.mixture_size >= 1 then 
        if prediction:size(2) == 2 then
            local rand_samples = torch.randn(m) -- generate m samples from standard normal distribution
            return torch.cmul(prediction[{{}, 2}], rand_samples) + prediction[{{}, 1}]
        else
            local n = prediction:size(2)/3
            local acc = torch.Tensor(m):zero()
            for i = 1, n do
                rand_samples = torch.randn(m)
                pred = torch.cmul(prediction[{{}, 2*n + i}], rand_samples) + prediction[{{}, n + i}]
                acc = acc + torch.cmul(prediction[{{}, i}], pred)
            end
            return acc
        end
    else
        local bin_size = 8/opt.nbins
        local probs = torch.exp(prediction)
        probs:div(torch.sum(probs))
        local bins = torch.multinomial(probs, 1):double() -- sample bins from softmax
        return ((bins - 1) * bin_size):squeeze() + torch.rand(m) * bin_size - 5 -- sample within bin
    end
end

local function plotTraj(x, states)
    t = torch.linspace(0, 9.9)
    gnuplot.pngfigure('./figures/d' .. x[{30, 1, 1}] .. '.png')
    gnuplot.plot({'True', states[{{22, 120}, 1}], '~'}, {'Simulated 1', x[{{22, 120}, 1, 2}], '~'}, 
        {'Simulated 2', x[{{22, 120}, 2, 2}], '~'}, {'Simulated 3', x[{{22, 120}, 3, 2}], '~'})
    gnuplot.plotflush()

    gnuplot.pngfigure('./figures/r' .. x[{30, 1, 1}] .. '.png')
    gnuplot.plot({'True', states[{{22, 120}, 2}], '~'}, {'Simulated 1', x[{{22, 120}, 1, 3}], '~'}, 
        {'Simulated 2', x[{{22, 120}, 2, 3}], '~'}, {'Simulated 3', x[{{22, 120}, 3, 3}], '~'})
    gnuplot.plotflush()

    gnuplot.pngfigure('./figures/vel' .. x[{30, 1, 1}] .. '.png')
    gnuplot.plot({'True', states[{{22, 120}, 3}], '~'}, {'Simulated 1', x[{{22, 120}, 1, 4}], '~'}, 
        {'Simulated 2', x[{{22, 120}, 2, 4}], '~'}, {'Simulated 3', x[{{22, 120}, 3, 4}], '~'})
    gnuplot.plotflush()

    gnuplot.pngfigure('./figures/acc' .. x[{30, 1, 1}] .. '.png')
    gnuplot.plot({'True', states[{{22, 120}, 4}], '~'}, {'Simulated 1', x[{{22, 120}, 1, 5}], '~'},
        {'Simulated 2', x[{{22, 120}, 2, 5}], '~'}, {'Simulated 3', x[{{22, 120}, 3, 5}], '~'})
    gnuplot.plotflush()
end

-- Function to generate velocity predictions at 1, ..., 10 sec horizons
-- using recurrent neural network
local function propagate(states, target, x_lead, s_lead, plot)

	-- Create tensor to hold state at each time step, starting 2 seconds back
	-- state, x = [d(t), r(t), s(t), a(t)]
	local x = torch.Tensor(121, m, 5):zero()

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

       -- Fill in states at t = -2 sec to t = 0
	   x[{{1, 21}, i, {2, 5}}] = states[{{1, 21}, {}}]
    end

    -- forward pass
    for t=1,120 do

        local lst = protos.rnn:forward{x[{t, {}, {2, 5}}], unpack(current_state)}
        -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
        current_state = {}
        for i=1,state_size do table.insert(current_state, lst[i]) end
        if t > 20 then -- Generate predictions after 2 sec
            local prediction = lst[#lst] -- last element holds the acceleration prediction
            local acc = sample(prediction)

        	x[{t+1, {}, 1}] = x[{t, {}, 1}] + torch.mul(x[{t, {}, 4}], 0.1) + torch.mul(acc, 0.01) -- x_ego(t + dt)
            x[{t+1, {}, 2}] = -x[{t+1, {}, 1}] + x_lead[t - 20] -- d(t + dt)
            x[{t+1, {}, 4}] = x[{t, {}, 4}] + torch.mul(x[{t, {}, 5}], 0.1) -- s(t + dt)
            x[{t+1, {}, 3}] = -x[{t+1, {}, 4}] + s_lead[t - 20] -- r(t + dt)
            x[{t+1, {}, 5}] = acc -- a(t + dt)
        end   
    end

    -- Return propagated trajectories
	if plot then plotTraj(x, states) end
end

-- Function to find the RWSE over different prediction horizons on the validation set
function analyze.findError(loader)

    -- Load inputs/targets
    local states = loader.X[loader.valSet]
    local target = loader.vel[loader.valSet]
    local x_lead = loader.x_lead[loader.valSet]
    local s_lead = loader.s_lead[loader.valSet]

    --Define # of trajectories to simulate
    m = 3

    -- Initialize network
    current_state = {}
    for L=1,opt.num_layers do
        local h_init = torch.zeros(m, opt.nn_size)
        if opt.gpuid >=0 then h_init = h_init:cuda() end
        table.insert(current_state, h_init:clone())
        table.insert(current_state, h_init:clone())
    end
    state_size = #current_state
    states = torch.reshape(states, loader.batches*opt.batch_size, 120, 4)
    protos.rnn:evaluate()

    target = torch.reshape(target, loader.batches*opt.batch_size, 100)
    x_lead = torch.reshape(x_lead, loader.batches*opt.batch_size, 100)
    s_lead = torch.reshape(s_lead, loader.batches*opt.batch_size, 100)

    -- Create tensor to hold errors over different horizons
    local err = torch.Tensor(states:size(1), 10)
    local size = 0

    -- Loop over all inputs
    for i = 1, states:size(1) do
        if target[i][100] ~= 0 then -- Avoid wasting time on abbreviated trajectories
            size = size + 1
            if size % 300 == 0 then
                propagate(states[i], target[i], x_lead[i], s_lead[i], true)
            else
                propagate(states[i], target[i], x_lead[i], s_lead[i], false)
            end
        end
        if i%500 == 0 then print(i) end
    end

    -- -- Resize err tensor
    -- err = err[{{1, size}, {}}]

    -- -- Find average error at each horizon, then take square root
    -- local RWSE = torch.mean(err, 1)
    -- RWSE = torch.sqrt(RWSE)

    
    -- print('RNN RWSE is: ')
    -- print(RWSE[1])

    collectgarbage()
    -- return RWSE
end

return analyze