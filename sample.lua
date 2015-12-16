local torch = require 'torch'
local nn = require 'nn'
require 'nngraph'
local MDN = require 'mdn'
local model = require 'model'
local model_utils = require 'model_utils'
local data = require 'data'

cmd = torch.CmdLine()
cmd:option('-model', 'model.t7','path to the input model file')
cmd:option('-seed', 123, 'random number generator seed')
cmd:option('-length', 1000, 'number of frames to generate')
cmd:option('-output', 'test.wav', 'path to the output audio file')
cmd:option('-bias', 1, 'sample variance scalar')
opt = cmd:parse(arg)

torch.manualSeed(opt.seed)

m = torch.load(opt.model)

local protos = m.model.protos
local lstm_state = model_utils.clone_list(m.model.initstate)
for i = 1, #lstm_state do
    lstm_state[i] = torch.Tensor(1, lstm_state[i]:size(2)):copy(lstm_state[i][{1, {}}])
end

-- seeding
m.input.current_batch = 0
local x, _ = data.next_batch(m.input)
for t = 0, m.params.seq_length * 10 do
    local i = (t % m.params.seq_length) + 1
    local next_state = protos.lstm:forward{
        torch.Tensor(1, x:size(3)):copy(x[{1, i, {}}]),
        unpack(lstm_state)
    }
    protos.linear_out:forward(next_state[#next_state])
    lstm_state = next_state
end

-- sequence generation
local sequence = {[0]=torch.zeros(1, m.params.input_size)}

for t = 1, opt.length do
    local next_state = protos.lstm:forward{
        torch.Tensor(1, x:size(3)):copy(sequence[t - 1]),
        unpack(lstm_state)
    }
    local probs = protos.linear_out:forward(next_state[#next_state])
    sequence[t] = protos.criterion:sample(probs, opt.bias):clone()
    lstm_state = next_state
end

data.synthesise(sequence, m.params, opt.output)
