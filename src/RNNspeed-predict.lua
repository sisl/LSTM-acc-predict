-- Script that creates a recurrent neural network for speed prediction
-- Largely based off of char-rnn by Andrej Karpathy
-- https://github.com/karpathy/char-rnn

require 'nn'
require 'optim'
require 'torch'
require 'nngraph'
require 'util.misc'
require 'normalNLL'
local CarDataLoader = require 'util.CarDataLoader'
local model_utils = require 'util.model_utils'
local LSTMnorm = require 'model.LSTMnorm'
local LSTM = require 'model.LSTM'
convert = require 'util.convert'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a recurrent neural network to generate speed predictions')
cmd:text()
cmd:text('Options')
-- model params
cmd:option('-nn_size', 64, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-mixture_size', 0, 'number of Gaussian mixtures in output layer')
cmd:option('-multinomial_size', 0, 'number of multinomial distributions')
cmd:option('-nbins', 0, 'number of bins if performing softmax')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-dropout',0,'dropout for regularization, used after each hidden layer. 0 = no dropout')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',3,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-nfolds',10,'number of folds to use in cross-validation')
cmd:option('-valSet', 1, 'fold to be held out as validation set')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',10,'number of sequences to train on in parallel')
cmd:option('-epochs', 1,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-seed',123,'torch manual random number generator seed')
-- saving network
cmd:option('-checkpoint_dir', 'nets', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-savenet', false, 'whether to save network parameters')
-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- initialize cunn/cutorch for training
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) 
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- Load data
local loader = CarDataLoader.create(opt.nfolds, opt.batch_size, true)

print('creating an LSTM with ' .. opt.num_layers .. ' layers')
assert(opt.mixture_size == 0 or opt.nbins == 0, 'must select only one method')

-- Specify number of outputs depending on distribution type
if opt.mixture_size == 1 then 
    outputs = 2
elseif opt.mixture_size > 1 then
    outputs = 3*opt.mixture_size
elseif opt.nbins > 1 then
    outputs = opt.nbins

    print('Finding bin boundaries...')
    bounds = convert.findBoundaries(loader, opt.nbins)
    print(bounds)
else
    error('no prediction method selected')
end
inputs = 4 -- size of input state
protos = {}


-- Set criterion, again depends on output type
if opt.mixture_size >= 1 and not opt.iter then
    protos.rnn = LSTMnorm.lstm(inputs, outputs, opt.nn_size, opt.num_layers, opt.dropout) -- LSTM with Gaussian mixture output
    protos.criterion = normalNLL(opt.mixture_size) 
else
    protos.rnn = LSTM.lstm(inputs, outputs, opt.nn_size, opt.num_layers, opt.dropout) -- LSTM with piecewise uniform output
    protos.criterion = nn.ClassNLLCriterion()
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.nn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor and initialize
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
params:uniform(-0.08, 0.08) -- small numbers uniform

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, 120, not proto.parameters)
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
-- Define function to evealuate the loss over a batch of 
-- training data
feval = function(params_new)

	-- Set params to copy of params_new if different (shouldn't be)
	if params_new ~= params then
        params:copy(params_new)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    -- Check that there are more training batches left
    assert(loader.moreBatches, 'No more batches -- something is wrong...')

    -- get next batch of inputs/targets
    local x, y = loader:next_batch()

    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        x = x:cuda()
        y = y:cuda()
    end

    prev_acc = torch.zeros(opt.batch_size)
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}  
    local loss = 0
    for t=1,120 do

        -- Set network to proper mode for training/evaluation
        if loader.val then
            clones.rnn[t]:evaluate()
        else
            clones.rnn[t]:training() 
        end

        local lst = clones.rnn[t]:forward{convert.augmentInput(x[{{}, t}], loader), unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        -- Initialize internal state with 2 sec of data; only calculate loss after this
        if t > 20 then
        	predictions[t] = lst[#lst] -- last element is the prediction
            if opt.mixture_size >= 1 then
        	    loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t - 20}])
            else
                loss = loss + clones.criterion[t]:forward(predictions[t], convert.toBins(y[{{}, t - 20}], opt.nbins))
            end
        end
    end
    loss = loss / 100

    -- Perform backprop if not evaluating loss on validation set
    if loader.val == false then 
        ------------------ backward pass -------------------
        -- initialize gradient at time t to be zeros (there's no influence from future)
        local drnn_state = {[120] = clone_list(init_state, true)} -- true also zeros the clones
        for t=120,21,-1 do
            -- backprop through loss
            if opt.mixture_size >= 1 then
                local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t - 20}])
                table.insert(drnn_state[t], doutput_t)
            else
                local doutput_t = clones.criterion[t]:backward(predictions[t], convert.toBins(y[{{}, t - 20}], opt.nbins))
                table.insert(drnn_state[t], doutput_t)
            end
            
            local dlst = clones.rnn[t]:backward({convert.augmentInput(x[{{}, t}], loader), unpack(rnn_state[t-1])}, drnn_state[t])
            drnn_state[t-1] = {}
            for k,v in pairs(dlst) do
                if k > 1 then -- k == 1 is gradient on x, which we dont need
                    -- then we do k-1 to start with derivative of states
                    drnn_state[t-1][k-1] = v
                end
            end
        end
        ------------------------ misc ----------------------
        -- transfer final state to initial state (BPTT)
        init_state_global = rnn_state[#rnn_state]
        -- clip gradient element-wise
        grad_params:clamp(-opt.grad_clip, opt.grad_clip)
        collectgarbage()
    end
    return loss, grad_params
end

-- Function to evaluate loss on validation set
function valLoss()
    print('Evaluating loss on validation set for fold ' .. loader.valSet .. '...')

    -- Evaluate loss on validation set
    -- Define batch indices
    loader.batch_ix = {loader.valSet, 0}
    loader.val = true
    loader.moreBatches = true

    val_loss = 0

    -- Iterate over all batches in all folds in validation set
    iterations = loader.batches
    for j = 1, iterations do
        local _, loss = optim.rmsprop(feval, params, rms_params)
        val_loss = val_loss + loss[1] -- the loss is inside a list, pop it
    end
    print(string.format('Average loss on validation set is %.3f', val_loss/iterations))
    return val_loss/iterations
end

-- Set options for rmsprop
rms_params = {
    learningRate = opt.learning_rate,
    alpha = opt.decay_rate
}

-- Define overall prediction horizon in sec
local train_loss = 0

-- Set validation set
loader.valSet = opt.valSet
print('Fold ' .. loader.valSet .. ' being used as validation set')

-- Initialize validation loss
local old_loss = 1e6

-- Loop through data for desired number of epochs
for i = 1, opt.epochs do

    -- Find validation loss; end training once validation loss has leveled off
    val_loss = valLoss()
    if val_loss - old_loss > -0.01 then break end
    old_loss = val_loss

	-- Reset batch indices
	loader.batch_ix = {1, 0}
    loader.val = false
	loader.moreBatches = true

    -- exponential learning rate decay
    if i >= opt.learning_rate_decay_after then
        local decay_factor = opt.learning_rate_decay
        rms_params.learningRate = rms_params.learningRate * decay_factor -- decay it
    end

	-- Iterate over all batches in all folds in training set
	iterations = (opt.nfolds - 1) * loader.batches
	for j = 1, iterations do

        -- Find loss for minibatch
		local timer = torch.Timer()
    	local _, loss = optim.rmsprop(feval, params, rms_params)
    	local time = timer:time().real
    	train_loss = train_loss + loss[1]

        -- Display loss
    	if (j + (i - 1)*iterations) % 10 == 0 then
			print(string.format("%d/%d: train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", 
				j + (i - 1)*iterations, opt.epochs*iterations, train_loss/10, grad_params:norm() / params:norm(), time))
            last_loss = train_loss/10
			train_loss = 0
		end
	end
    collectgarbage()
end

-- Save network parameters
if opt.savenet then
    local savefile = string.format('%s/%s_epochs%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, opt.epochs, last_loss)
    print('saving checkpoint to ' .. savefile)
    local checkpoint = {}
    checkpoint.protos = protos
    checkpoint.opt = opt
    torch.save(savefile, checkpoint)
end




