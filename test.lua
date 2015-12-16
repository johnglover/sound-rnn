local nn = require 'nn'
local mdn = require 'mdn'
local model = require 'model'
local model_utils = require 'model_utils'

torch.manualSeed(123)

local test_runner = torch.Tester()
local test_suite = {}
local precision = 1e-2  -- used for checking relative error
local eps = 1e-6

local function criterionJacobianTest(cri, input, target)
    local an_grad = cri:backward(input, target)
    local num_grad = torch.Tensor():resizeAs(input)

    for batch = 1, input:size(1) do
        for i = 1, input:size(2) do
            local x = torch.Tensor(1, input:size(2)):copy(input[batch])
            local y = torch.Tensor(1, target:size(2)):copy(target[batch])
            x[1][i] = x[1][i] + eps
            local fx1 = cri:forward(x, y)
            x[1][i] = x[1][i] - 2 * eps
            local fx2 = cri:forward(x, y)
            num_grad[batch][i] = (fx1 - fx2) / (2 * eps)
            x[1][i] = x[1][i] + eps
        end
    end

    local err = (num_grad - an_grad):cdiv(num_grad):abs():max()
    test_runner:assertlt(err, precision, 'error in relative difference between numerical and analytical gradients')
end

function test_suite.mdn_gradient()
    local mdn_components = 5
    local num_dims = 10
    local input_size = (2 * mdn_components * num_dims) + mdn_components
    local batch_size = 5

    local input = torch.rand(batch_size, input_size)
    local target = torch.rand(batch_size, num_dims)
    local cri = nn.MDNCriterion(mdn_components)
    criterionJacobianTest(cri, input, target)
end

local function compute_loss(params, input, target, mod, seq_length)
    if params ~= mod.params then
        mod.params:copy(params)
    end

    local lstm_state = {[0]=mod.initstate}
    local predictions = {}
    local loss = 0

    for t = 1, seq_length do
        lstm_state[t] = mod.clones.lstm[t]:forward{input, unpack(lstm_state[t - 1])}
        predictions[t] = mod.clones.linear_out[t]:forward(lstm_state[t][#lstm_state[t]])
        loss = loss + mod.clones.criterion[t]:forward(predictions[t], target)
    end

    return loss
end

local function compute_gradient(input, target, mod, seq_length)
    mod.grad_params:zero()

    local lstm_state = {[0]=mod.initstate}
    local predictions = {}
    local loss = 0

    for t = 1, seq_length do
        lstm_state[t] = mod.clones.lstm[t]:forward{input, unpack(lstm_state[t - 1])}
        predictions[t] = mod.clones.linear_out[t]:forward(lstm_state[t][#lstm_state[t]])
        loss = loss + mod.clones.criterion[t]:forward(predictions[t], target)
    end

    local dlstm_state = {[seq_length] = model_utils.clone_list(mod.initstate, true)}

    for t = seq_length, 1, -1 do
        local doutput = mod.clones.criterion[t]:backward(predictions[t], target)

        if t == seq_length then
            dlstm_state[t][#dlstm_state[t]] = mod.clones.linear_out[t]:backward(lstm_state[t][#lstm_state[t]], doutput)
        else
            dlstm_state[t][#dlstm_state[t]]:add(mod.clones.linear_out[t]:backward(lstm_state[t][#lstm_state[t]], doutput))
        end

        local dlstm = mod.clones.lstm[t]:backward({input, lstm_state[t - 1]}, dlstm_state[t])
        dlstm_state[t - 1] = {}
        for k, v in pairs(dlstm) do
            if k > 1 then
                dlstm_state[t - 1][k - 1] = v
            end
        end
    end

  return mod.grad_params
end

function test_suite.test_model()
    local input_size = 10
    local rnn_size = 20
    local num_layers = 1
    local mdn_components = 5
    local seq_length = 1
    local batch_size = 1

    local m = model.new(
        input_size,
        rnn_size,
        num_layers,
        mdn_components,
        seq_length,
        batch_size
    )
    local input = torch.rand(batch_size, input_size)
    local target = torch.rand(batch_size, input_size)

    local an_grad = compute_gradient(input, target, m, seq_length)

    local x = m.params
    local num_grad = torch.DoubleTensor(an_grad:size())
    for i = 1, an_grad:size(1) do
        x[i] = x[i] + eps
        local fx1 = compute_loss(x, input, target, m, seq_length)
        x[i] = x[i] - 2 * eps
        local fx2 = compute_loss(x, input, target, m, seq_length)
        num_grad[i] = (fx1 - fx2) / (2 * eps)
    end

    local err = (num_grad - an_grad):cdiv(num_grad):abs():max()
    test_runner:assertlt(err, precision, 'error in relative difference between numerical and analytical gradients')
end

test_runner:add(test_suite)
test_runner:run()
