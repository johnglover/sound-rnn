local torch = require 'torch'
local nn = require 'nn'
require 'nngraph'
local MDN = require 'mdn'
local model_utils = require 'model_utils'

local model = {}

-- Long Short-Term Memory
-- from: https://github.com/karpathy/char-rnn
function lstm(input_size, rnn_size, n, dropout)
    dropout = dropout or 0

    -- 2 * n + 1 inputs
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- x
    for L = 1, n do
        table.insert(inputs, nn.Identity()()) -- prev_c[L]
        table.insert(inputs, nn.Identity()()) -- prev_h[L]
    end

    local x, input_size_L
    local outputs = {}

    for L = 1, n do
        -- c, h from previos timesteps
        local prev_c = inputs[L * 2]
        local prev_h = inputs[L * 2 + 1]

        -- the input to this layer
        if L == 1 then
            x = inputs[1]
            input_size_L = input_size
        else
            x = outputs[(L - 1) * 2]
            if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
            input_size_L = rnn_size
        end

        -- evaluate the input sums at once for efficiency
        local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_' .. L}
        local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_' .. L}
        local all_input_sums = nn.CAddTable()({i2h, h2h})

        local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
        local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)

        -- decode the gates
        local in_gate = nn.Sigmoid()(n1)
        local forget_gate = nn.Sigmoid()(n2)
        local out_gate = nn.Sigmoid()(n3)

        -- decode the write inputs
        local in_transform = nn.Tanh()(n4)

        -- perform the LSTM update
        local next_c = nn.CAddTable()({
            nn.CMulTable()({forget_gate, prev_c}),
            nn.CMulTable()({in_gate, in_transform})
        })

        -- gated cells form the output
        local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

        table.insert(outputs, next_c)
        table.insert(outputs, next_h)
    end

    return nn.gModule(inputs, outputs)
end

function model.new(input_size, rnn_size, num_layers, mdn_components, seq_length, batch_size, gpu_id)
    local mdn_size = (2 * mdn_components * input_size) + mdn_components

    -- define model prototypes for 1 timestep
    local protos = {}
    protos.lstm = lstm(input_size, rnn_size, num_layers)
    protos.linear_out = nn.Linear(rnn_size, mdn_size)
    protos.criterion = nn.MDNCriterion(mdn_components)
    
    -- ship protos to GPU if needed
    if gpu_id >= 0 then
       print('Shipping model to GPU')
       for k, v in pairs(protos) do 
          print('Shipping ' .. k)
          v:cuda()
       end
    end

    -- combine parameters into one flattened parameters tensor
    local params, grad_params = model_utils.combine_all_parameters(protos.lstm, protos.linear_out)

    -- initialise parameters
    params:uniform(-0.08, 0.08)

    -- initialize the LSTM forget gates with higher biases to encourage remembering
    for L = 1, num_layers do
        for _, node in ipairs(protos.lstm.forwardnodes) do
            if node.data.annotations.name == "i2h_" .. L then
                -- the gates are, in order, i, f, o, g, so f is the 2nd block of weights
                node.data.module.bias[{{rnn_size + 1, 2 * rnn_size}}]:fill(1.0)
            end
        end
    end

    print(string.format('creating model (%d parameters)', params:size(1)))
        -- make clones (after flattening, as that reallocates memory)
    local clones = {}
    for name, proto in pairs(protos) do
        print('cloning ' .. name)
        clones[name] = model_utils.clone_many_times(proto, seq_length)
    end

    local new_model = {}
    new_model.protos = protos
    new_model.params = params
    new_model.grad_params = grad_params
    new_model.clones = clones

    new_model.initstate = {}
    new_model.dfinalstate = {}
    for i = 1, num_layers do
        local h_init = torch.zeros(batch_size, rnn_size)
        if gpu_id >= 0 then h_init = h_init:cuda() end
        table.insert(new_model.initstate, h_init) -- c
        table.insert(new_model.initstate, h_init) -- h
        table.insert(new_model.dfinalstate, h_init) -- c
    end

    return new_model
end

return model
