-- LSTM model with output layer containing parameters to a Gaussian mixture
local LSTM = {}
function LSTM.lstm(input_size, output_size, rnn_size, n, dropout)
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previous timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})
    -- decode the gates
    local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)
    -- decode the write inputs
    local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, output_size)(top_h)


  if output_size == 2 then -- Normal distribution
    local mu = nn.Narrow(2, 1, 1)(proj) -- mean

    local pre_exp = nn.Narrow(2, 2, 1)(proj) -- std dev (enforce positive)
    local sigma = nn.Exp()(pre_exp)
    
    local join = nn.JoinTable(2)({mu, sigma})
    local out = nn.Reshape(2)(join)
    table.insert(outputs, out)
    
  else
    local n = output_size/3 -- Gaussian mixture
    
    local pre_w = nn.Narrow(2, 1, n)(proj) -- First n are weights -> apply softmax
    local w = nn.SoftMax()(pre_w)

    local mu = nn.Narrow(2, n + 1, n)(proj) -- next n are mu
    
    local pre_exp = nn.Narrow(2, 2*n + 1, n)(proj) -- last n are sigma, apply exponential
    local sigma = nn.Exp()(pre_exp)

    local join = nn.JoinTable(2)({w, mu, sigma}) -- join and reshape
    local out = nn.Reshape(3*n)(join)
    table.insert(outputs, out)
  end

  return nn.gModule(inputs, outputs)
end

return LSTM

