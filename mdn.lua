-- Implementation of a Mixture Density Network loss Criterion.
-- Also includes a function for sampling from the PDF.
--
-- For more on Mixture Density Networks, see:
-- Bishop ("Mixture Density Networks"): http://eprints.aston.ac.uk/373/1/NCRG_94_004.pdf
-- Schuster ("On supervised learning from sequential data with applications for speech recognition"): http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.17.1460&rep=rep1&type=pdf
-- The latter contains the gradient functions for the multivariate case.

local torch = require 'torch'
local nn = require 'nn'
require 'nngraph'
local rk = require 'randomkit'

local eps = 1e-12

local MDN, Parent = torch.class('nn.MDNCriterion', 'nn.Criterion')

local function num_dims(self, input)
    if input:dim() ~= 2 then
        error('input must be 2-dimensional.')
    end

    return (input:size(2) - self.num_components) / (2 * self.num_components)
end

local function get_params(self, input)
    local x = self.input_buffer:resizeAs(input):copy(input)
    local dims = num_dims(self, input)
    local Nc = self.num_components

    -- first num_components values are mixture weights
    local weights = x.new(x:size(1), Nc)
    weights:copy(x[{{}, {1, Nc}}])
    weights:exp():cdiv(weights:sum(2):expandAs(weights))

    -- next (num_components * num_vars) values are means
    local means = x[{{}, {Nc + 1, Nc + (Nc * dims)}}]
        :contiguous()
        :view(x:size(1), dims, Nc)

    -- remaining (num_components * num_vars) values are the variances
    vars = x.new(x:size(1), dims, Nc)
    vars:copy(
        x[{{}, {x:size(2) - (Nc * dims) + 1, x:size(2)}}]
    ):exp()

    return weights, means, vars
end

local function pdf(y, means, vars, weights, dims)
    local norm = vars:clone()
                     :prod(2)
                     :mul(math.pow(2 * math.pi, dims / 2))

    return y:clone()
            :add(-means)
            :cdiv(vars)
            :pow(2)
            :sum(2)
            :mul(-0.5)
            :exp()
            :cdiv(norm)
            :cmul(weights)
end

-- If any of the responsibilites are 0, set them to 1 / num_components,
-- corresponding to an unconditional uniform prior that the vector was generated
-- by any mixture component.
local function responsibilities(probs, num_components)
    local norm = probs:sum(3):apply(function(x)
        local y = x
        if x < 1e-50 then
            y = 1 / num_components
        end
        return y
    end)
    return torch.cdiv(probs, norm:expandAs(probs))
end

function MDN:__init(num_components)
    Parent.__init(self)
    self.num_components = num_components
    self.input_buffer = torch.Tensor()
    self.gradInput = torch.Tensor()
    self.sample_buffer = torch.Tensor()
end

function MDN:cuda()
    print('Shipping MDN to GPU')
    Parent.cuda(self)
    self.input_buffer = self.input_buffer:cuda()
    self.gradInput = self.gradInput:cuda()
    self.sample_buffer = self.sample_buffer:cuda()
end

function MDN:updateOutput(input, target)
    local dims = num_dims(self, input)
    local y = target:resize(target:size(1), dims, 1)
                    :expand(target:size(1), dims, self.num_components)

    local weights, means, vars = get_params(self, input)
    local probs = pdf(y, means, vars, weights, dims)

    return -math.log(probs:sum() + eps)
end

function MDN:updateGradInput(input, target)
    local grad = self.gradInput:resizeAs(input)
    local dims = num_dims(self, input)
    local Nc = self.num_components
    local y = target:resize(target:size(1), dims, 1)
                    :expand(target:size(1), dims, Nc)

    local weights, means, vars = get_params(self, input)
    local probs = pdf(y, means, vars, weights, dims)
    local pi = responsibilities(probs, Nc)

    local dweights = torch.add(weights, -pi)
    local dmeans = means:clone()
                        :add(-y)
                        :cdiv(vars:clone():pow(2))
                        :cmul(pi:expandAs(y))
    local dvars = y:clone()
                   :add(-means)
                   :cdiv(vars)
                   :pow(2)
                   :add(-1)
                   :cmul(-pi:expandAs(y))

    grad[{{}, {1, Nc}}]:copy(dweights)
    grad[{{}, {Nc + 1, Nc + (Nc * dims)}}]:copy(dmeans:view(target:size(1) * dims * Nc))
    grad[{{}, {grad:size(2) - (Nc * dims) + 1, grad:size(2)}}]:copy(dvars)

    return grad
end

function MDN:sample(input, bias)
    local dims = num_dims(self, input)
    local output = self.sample_buffer:resize(1, dims):zero()
    local weights, means, vars = get_params(self, input)
    vars:mul(bias)
    local diag = torch.eye(dims)

    -- sample from component with largest weight
    local _, i = torch.max(weights, 2)
    i = i[1][1]
    local chol = torch.potrf(
        diag:clone()
            :cmul(vars[1][{{}, i}]:contiguous():view(1, dims):expandAs(diag))
    )
    local z = rk.standard_normal(torch.Tensor(dims))
    output:add(means[1]:select(2, i):clone():add(torch.mv(chol, z)))

    return output
end

return MDN
