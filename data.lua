local torch = require 'torch'
local nn = require 'nn'
require 'nngraph'
require 'optim'
require 'audio'
local signal = require 'signal'
local pv = require 'phase_vocoder'

local data = {}

local function rotate(x)
    local y = x:clone()
    y:sub(1, -2):copy(x:sub(2, -1))
    y[-1] = x[1]
    return y
end

function data.select_frame(frames, num_partials, randomise)
    local N = frames:size(2) * frames:size(3)
    local Np = num_partials * 2
    local frame = math.floor(frames:size(1) / 2)
    if randomise then
        local frame = math.random(math.floor(frames:size(1) / 4),
                                  math.floor((frames:size(1) * 3) / 4))
    end
    return torch.Tensor(Np):copy((frames[frame]:view(N, -1)[{{1, Np}}]))
end

function data.preprocess(audio, params)
    local data = {}
    local frames = pv.pva(audio, params.fft_size, params.fft_size / 4, params.sample_rate)

    print('preparing audio data')

    -- remove the end of the frames so that there is an even number of batches/sequences
    if frames:size(1) % (params.batch_size * params.seq_length) ~= 0 then
        frames = frames:sub(
            1,
            params.batch_size * params.seq_length * math.floor(
                frames:size(1) / (params.batch_size * params.seq_length)
            )
        )
    end

    -- flatten the 2D (amp, freq) frames to a 1D array, normalise and scale values,
    -- and limit the number of components
    local N = frames:size(2) * frames:size(3)
    local Np = params.num_partials * 2
    local x = torch.Tensor(frames:size(1), Np):zero()
    params.input_size = Np

    -- store values that can be used to initialise the model when generating sequences
    params.initial_frame = torch.Tensor(Np):copy((frames[5]:view(N, -1)[{{1, Np}}]))
    local deltas = rotate(frames):add(-frames)

    local amps = deltas:select(3, 1)[{{}, {1, params.num_partials}}]
    params.amp_mean = amps:mean()
    params.amp_std = amps:std()
    amps:add(-params.amp_mean)
    amps:div(params.amp_std)

    local freqs = deltas:select(3, 2)[{{}, {1, params.num_partials}}]
    params.freq_mean = freqs:mean()
    params.freq_std = freqs:std()
    freqs:add(-params.freq_mean)
    freqs:div(params.freq_std)

    for i = 1, deltas:size(1) do
        x[i]:copy(deltas[i]:view(N, -1)[{{1, Np}}])
    end

    local y = rotate(x)
    data.x = x:view(params.batch_size, frames:size(1) / params.batch_size, Np)
              :split(params.seq_length, 2)
    data.y = y:view(params.batch_size, frames:size(1) / params.batch_size, Np)
              :split(params.seq_length, 2)
    data.num_batches = #data.x

    collectgarbage()
    return data
end

function data.postprocess(input, params, prev_frame)
    local N = input:size(1)
    local output = torch.Tensor(N / 2, 2):zero()

    output:select(2, 1):copy(
        input:clone()
             :view(N / 2, -1)
             :select(2, 1)
             :mul(params.amp_std)
             :add(params.amp_mean)
    )

    output:select(2, 2):copy(
        input:clone()
             :view(N / 2, -1)
             :select(2, 2)
             :mul(params.freq_std)
             :add(params.freq_mean)
    )

    output:add(prev_frame)

    return output
end

function data.load_file(file_path)
    local x = audio.load(file_path)
    x:div(math.pow(2, 32)):type('torch.DoubleTensor')
    return x
end

function data.next_batch(data)
    if not data.current_batch then
        data.current_batch = 0
    end

    data.current_batch = (data.current_batch % data.num_batches) + 1
    return data.x[data.current_batch], data.y[data.current_batch]
end

function data.synthesise(deltas, params, file_path)
    local num_partials = params.input_size / 2
    local frames = torch.Tensor(#deltas, num_partials, 2)
    local prev = params.initial_frame:clone()
    for i = 1, #deltas do
        frames[i]:copy(data.postprocess(deltas[i][1], params, prev))
        prev:copy(frames[i])
    end
    local synth = pv.pvs(frames, num_partials, params.fft_size, params.fft_size / 4, params.sample_rate)

    -- raising this to the power of 31 here to prevent possible clipping, this could be a bit
    -- more sophisticated...
    synth:mul(math.pow(2, 31)):floor():type('torch.IntTensor')
    audio.save(file_path, synth, params.sample_rate)
end

return data
