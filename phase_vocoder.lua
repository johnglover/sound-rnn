local signal = require 'signal'

local phase_vocoder = {}

function phase_vocoder.pva(input_signal, fft_size, hop_size, sample_rate)
    print('performing phase vocoder analysis')

    local x = input_signal:clone()

    -- zero pad input signal if it doesn't divide evenly into frames
    if x:size()[1] % hop_size ~= 0 then
        x:resize(x:size()[1] + (hop_size - (x:size()[1] % hop_size)))
    end

    local window = signal.hann(fft_size)
    local num_bins = (fft_size / 2) + 1
    local num_frames = (x:size()[1] / hop_size) - (fft_size / hop_size)
    local last_phase = torch.Tensor(num_bins)
    local two_pi = 2 * math.pi
    local bin_centre_freq_scalar = (two_pi * hop_size) / fft_size
    local radians_to_hz_scalar = sample_rate / (two_pi * hop_size)
    local output = torch.Tensor(num_frames, num_bins, 2):zero()

    for i = 0, num_frames - 1 do
        -- window frame
        local offset = i * hop_size
        local windowed = x[{{1 + offset, offset + fft_size}}]:clone():cmul(window)

        -- rotate frame
        local rotated = torch.Tensor(fft_size)
        local m = offset % fft_size
        for j = 1, fft_size do
            rotated[((j - 1 + m) % fft_size) + 1] = windowed[j]
        end

        local fft = signal.fft(rotated)[{{1, num_bins}}]

        local amps = signal.complex.abs(fft)
        local phases = signal.complex.angle(fft)

        if i == 0 then
            last_phase:copy(phases)
        end

        for k = 2, num_bins do
            -- get phase difference, wrap to [-pi, pi] range
            local delta = phases[k] - last_phase[k]
            while delta > math.pi do delta = delta - two_pi end
            while delta < -math.pi do delta = delta + two_pi end

            -- output amplitudes and frequencies
            output[i + 1][k][1] = amps[k]
            output[i + 1][k][2] = (delta + (k - 1) * bin_centre_freq_scalar) * radians_to_hz_scalar

            last_phase[k] = phases[k]
        end
    end

    return output
end

function phase_vocoder.pvs(frames, num_partials, fft_size, hop_size, sample_rate)
    print('performing phase vocoder synthesis')

    local num_frames = frames:size()[1]
    local two_pi = 2 * math.pi
    local window = signal.hann(fft_size)
    local num_bins = (fft_size / 2) + 1
    local sr_over_fft = sample_rate / fft_size
    local hz_to_radians_scalar = (two_pi * hop_size) / sample_rate
    local current_frame = torch.zeros(num_bins, 2)
    local phases = torch.zeros(fft_size)
    local output = torch.Tensor((num_frames * hop_size) + fft_size, 1):zero()

    for i = 1, num_frames - 1 do
        for k = 2, num_partials do
            local delta = (frames[i][k][2] - ((k - 1) * sr_over_fft)) * hz_to_radians_scalar
            local phi = phases[k] + delta
            local amp = frames[i][k][1]
            current_frame[k][1] = amp * math.cos(phi)
            current_frame[k][2] = amp * math.sin(phi)
            phases[k] = phi
        end

        local samples = signal.irfft(current_frame)

        for n = 1, fft_size do
            local p = ((i - 1) * hop_size) + n
            local m = ((i - 1) * hop_size) % fft_size
            output[p] = output[p] + samples[((n + m) % fft_size) + 1] * window[n]
        end
    end

    return output
end

return phase_vocoder
