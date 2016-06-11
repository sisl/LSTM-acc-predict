require 'nn'
require 'nngraph'
require 'optim'
require 'torch'
require 'normalNLL'
local CarDataLoader = require 'util.CarDataLoader'
local convert = require 'util.convert'
local FFGM = require 'model.FFGM'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a feedforward neural network to generate speed predictions')
cmd:text()
cmd:text('Options')
-- model params
cmd:option('-nn_size', 64, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-nbins', 0, 'number of bins if performing softmax')
cmd:option('-time_steps', 1, 'number of time steps worth of state values to pass as input')
cmd:option('-mixture_size', 0, 'number of time steps worth of state values to pass as input')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-dropout',0,'dropout for regularization, used after each hidden layer. 0 = no dropout')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',3,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-nfolds',10,'number of folds to use in cross-validation')
cmd:option('-valSet', 1, 'fold to be held out as validation set')
cmd:option('-dropout',0,'dropout for regularization, used after each hidden layer. 0 = no dropout')
cmd:option('-batch_size',10,'number of sequences to train on in parallel')
cmd:option('-epochs', 1,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-seed',123,'torch manual random number generator seed')
-- saving network
cmd:option('-checkpoint_dir', 'nets', 'output directory where checkpoints get written')
cmd:option('-savefile','ffnn_reconst','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-savenet', false, 'whether to save network parameters')
-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
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

if opt.mixture_size == 1 then 
    outputs = 2
elseif opt.mixture_size > 1 then
    outputs = 3*opt.mixture_size
elseif opt.nbins > 0 then
    outputs = opt.nbins
else
    error('no prediction method selected')
end
inputs = opt.time_steps * 4

-- Construct neural network
if opt.nbins > 0 then
    model = nn.Sequential()
    HUs = opt.nn_size
    model:add(nn.Linear(inputs, HUs))
    model:add(nn.ReLU())
    for i = 2, opt.num_layers do
	   model:add(nn.Linear(HUs, HUs))
	   model:add(nn.ReLU())
        model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(HUs, outputs))
    model:add(nn.LogSoftMax())

    -- Set loss criterion
    criterion = nn.ClassNLLCriterion()
else
    -- Construct feedforward Gaussian mixture model
    model = FFGM.ffgm(inputs, outputs, opt.nn_size, opt.dropout)
    criterion = normalNLL(opt.mixture_size)
end

-- Get network parameters
params, grad_params = model:getParameters()
params:uniform(-0.08, 0.08) -- small numbers uniform

-- Define function to evealuate the loss over a batch of 
-- training data
feval = function(params_new)

	-- Set params to copy of params_new if different (shouldn't be)
	if params_new ~= params then
        params:copy(params_new)
    end
    grad_params:zero()

    -- Check that there are more training batches left
    assert(loader.moreBatches, 'No more batches -- something is wrong...')

    -- get next batch of inputs/targets
    local x, y = loader:next_batch()
	local loss = 0

	if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:cuda()
        y = y:cuda()
    end

    -- Set network to proper mode for training/evaluation
    if loader.val then
        model:evaluate()
    else
        model:training() 
    end

    for t = 21, 120 do

		-- Evaluate loss and gradients
        if opt.mixture_size >= 1 then
            loss = loss + criterion:forward(model:forward(convert.augmentFF(x, t, loader)), y[{{}, t - 20}])
            model:backward(convert.augmentFF(x, t, loader), criterion:backward(model.output, y[{{}, t - 20}]))
        else
    	    loss = loss + criterion:forward(model:forward(convert.augmentFF(x, t, loader)), convert.toBins(y[{{}, t - 20}], opt.nbins))
  		    model:backward(convert.augmentFF(x, t, loader), criterion:backward(model.output, convert.toBins(y[{{}, t - 20}], opt.nbins)))
        end
  	end
  	loss = loss/100
  	grad_params:clamp(-opt.grad_clip, opt.grad_clip)
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
horizon = 10
local train_loss = 0

loader.valSet = opt.valSet
print('Fold ' .. loader.valSet .. ' being used as validation set')

-- Initialize validation loss
local old_loss = 1e6

-- Loop for desired number of epochs
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
        assert(opt.learning_rate_decay < 1, 'Learning rate decay will cause it to grow')
        local decay_factor = opt.learning_rate_decay
        rms_params.learningRate = rms_params.learningRate * decay_factor -- decay it
    end

	-- Iterate over all batches in all folds in training set
	iterations = (opt.nfolds - 1) * loader.batches
	for j = 1, iterations do

		local timer = torch.Timer()
    	local _, loss = optim.rmsprop(feval, params, rms_params)
    	local time = timer:time().real

    	train_loss = train_loss + loss[1] -- get loss

    	if (j + (i - 1)*iterations) % 200 == 0 then
			print(string.format("%d/%d: train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", 
				j + (i - 1)*iterations, opt.epochs*iterations, train_loss/200, grad_params:norm() / params:norm(), time))
                last_loss = train_loss/200
			train_loss = 0
		end
	end
end

-- Save network parameters
if opt.savenet then
    local savefile = string.format('%s/%s_epochs%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, opt.epochs, last_loss)
    print('saving checkpoint to ' .. savefile)
    local checkpoint = {}
    checkpoint.model = model
    checkpoint.opt = opt
    torch.save(savefile, checkpoint)
end








