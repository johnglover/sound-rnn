local torch = require 'torch'
local nn = require 'nn'
require 'nngraph'
require 'optim'
local signal = require 'signal'
local data = require 'data'
local MDN = require 'mdn'
local model = require 'model'
local model_utils = require 'model_utils'

local cmd = torch.CmdLine()
cmd:option('-audio', 'audio.wav', 'path to the input audio file (monophonic)')
cmd:option('-model', 'model.t7', 'path to the output model file')
cmd:option('-batch_size', 1, 'number of sequences to train on in parallel')
cmd:option('-seq_length', 30, 'number of timesteps to unroll to')
cmd:option('-rnn_size', 100, 'number of LSTM units per layer')
cmd:option('-num_layers', 1, 'number of LSTM layers')
cmd:option('-mdn_components', 20, 'number of mixture components')
cmd:option('-fft_size', 512, 'FFT size')
cmd:option('-num_partials', 40, 'number of Sinusoidal components')
cmd:option('-sample_rate', 44100, 'the audio sample rate')
cmd:option('-max_epochs', 1000, 'number of full passes through the training data')
cmd:option('-model_file', 'model_autosave.t7', 'filename to autosave the model (protos) to')
cmd:option('-save_every', 100, 'save every 100 steps, overwriting the existing file')
cmd:option('-print_every', 100, 'how many steps/minibatches between printing out the loss')
cmd:option('-seed', 123, 'torch manual random number generator seed')
cmd:option('-learning_rate_decay', 0.97, 'learning rate decay')
cmd:option('-num_threads', 1, 'number of CPU threads to use')
cmd:option('-gpu_id', -1, 'ID of the GPU to use (-1 for CPU)')
local params = cmd:parse(arg)

torch.manualSeed(params.seed)
torch.setnumthreads(params.num_threads)

if params.gpu_id >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. params.gpu_id .. '...')
        cutorch.setDevice(params.gpu_id + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(params.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        params.gpu_id = -1 -- overwrite user setting
    end
end

local raw_audio = data.load_file(params.audio)
local input = data.preprocess(raw_audio, params)

local mod = model.new(
    params.input_size,
    params.rnn_size,
    params.num_layers,
    params.mdn_components,
    params.seq_length,
    params.batch_size,
    params.gpu_id
)

function feval(params_)
    if params_ ~= mod.params then
        mod.params:copy(params_)
    end
    mod.grad_params:zero()

    local x, y = data.next_batch(input)

    if params.gpu_id >= 0 then
      x = x:cuda()
      y = y:cuda()
    end

    -- forward pass
    local lstm_state = {[0] = mod.initstate}
    local predictions = {}
    local loss = 0

    for t = 1, params.seq_length do
        lstm_state[t] = mod.clones.lstm[t]:forward{x[{{}, t, {}}], unpack(lstm_state[t - 1])}
        predictions[t] = mod.clones.linear_out[t]:forward(lstm_state[t][#lstm_state[t]])
        loss = loss + mod.clones.criterion[t]:forward(predictions[t], y[{{}, t, {}}])
    end

    -- backward pass
    local dlstm_state = {[params.seq_length] = model_utils.clone_list(mod.initstate, true)}

    for t = params.seq_length, 1, -1 do
        local doutput = mod.clones.criterion[t]:backward(predictions[t], y[{{}, t, {}}])

        -- Two cases for dloss/dh_t:
        --   1. h_t is only used once, sent to the criterion (but not to the next LSTM timestep).
        --   2. h_t is used twice, for the criterion and for the next step. To obey the
        --      multivariate chain rule, we add them.
        if t == params.seq_length then
            dlstm_state[t][#dlstm_state[t]] = mod.clones.linear_out[t]:backward(lstm_state[t][#lstm_state[t]], doutput)
        else
            dlstm_state[t][#dlstm_state[t]]:add(mod.clones.linear_out[t]:backward(lstm_state[t][#lstm_state[t]], doutput))
        end

        local dlstm = mod.clones.lstm[t]:backward({x[{{}, t, {}}], unpack(lstm_state[t - 1])}, dlstm_state[t])
        dlstm_state[t - 1] = {}
        for k, v in pairs(dlstm) do
            if k > 1 then
                dlstm_state[t - 1][k - 1] = v
            end
        end
    end

    -- transfer final state to initial state (BPTT)
    mod.initstate = lstm_state[#lstm_state]

    -- clip gradient element-wise
    mod.grad_params:clamp(-5, 5)

    return loss, mod.grad_params
end

-- optimization
print('starting optimisation')
local losses = {}
local optim_state = {learningRate = 0.001}
for i = 1, params.max_epochs * input.num_batches do
    local _, loss = optim.adagrad(feval, mod.params, optim_state)

    losses[#losses + 1] = loss[1]

    if i % params.save_every == 0 then
        torch.save(params.model_file, {
            model = mod,
            input = input,
            params = params,
            losses = losses
        })
        collectgarbage()
    end

    if i % params.print_every == 0 then
        print(string.format(
            "iteration %4d, loss = %6.8f, gradnorm = %6.3e", i, loss[1], mod.grad_params:norm() / mod.params:norm()
        ))
    end

    if (params.learning_rate_decay < 1) and (i % 10000 == 0) then
        optim_state.learningRate = optim_state.learningRate * params.learning_rate_decay -- decay it
        print(string.format('learning rate = %.8f', optim_state.learningRate))
    end
end
