local th = require('torch')
local nn = require 'nn'
require 'nngraph'
require 'pl'
require '../config.lua'
require '../main.lua'

if arg.useCuda then
	require "cunn"
end

th.setdefaulttensortype('torch.FloatTensor')
--[[
    TODO:
    - Check if batchnorm parameters are correct
--]]
local function conv_layer(stream, nInputFeatMap, nOutputFeatMap)

    --Must make sure if stream[i] represents the weights of the very last layer!
    stream[1] = nn.SpatialConvolution(nInputFeatMap, nOutputFeatMap, 3,3, 1,1, 1,1)(stream[1])
    for i=2,arg.numstreams do
        stream[i] = nn.SpatialConvolution(nInputFeatMap, nOutputFeatMap, 3,3, 1,1, 1,1)
                    :clone(stream[i-1], "weight", "bias", "gradWeight", "gradBias")(stream[i])
    end

    stream[1] = nn.SpatialBatchNormalization(nOutputFeatMap, nil, 0.9)(stream[1])
    for i=2, arg.numstreams do
        stream[i] = nn.SpatialBatchNormalization(nOutputFeatMap, nil, 0.9)
                    :clone(stream[i-1], "weight", "bias", "gradWeight", "gradBias")(stream[i])
    end

    stream[1] = nn.ReLU()(stream[1])
    for i=2, arg.numstreams do
        stream[i] = nn.ReLU()
                    :clone(stream[i-1], "weight", "bias", "gradWeight", "gradBias")(stream[i])
    end

    return stream
end


local function local_layer(stream, dropout)

    stream[1] = nn.SpatialConvolutionLocal(64, 64, 16, 8, 1,1)(stream[1])
    for i=2, arg.numstreams do
        stream[i] = nn.SpatialConvolutionLocal(64, 64, 16, 8, 1,1)
                    :clone(stream[i-1], "weight", "bias", "gradWeight", "gradBias")(stream[i])
    end

    stream[1] = nn.SpatialBatchNormalization(64, nil, 0.9)(stream[1])
    for i=2, arg.numstreams do
        stream[i] = nn.SpatialBatchNormalization(64, nil, 0.9)
                    :clone(stream[i-1], "weight", "bias", "gradWeight", "gradBias")(stream[i])
    end

    stream[1] = nn.ReLU()(stream[1])
    for i=2, arg.numstreams do
        stream[i] = nn.ReLU()
                    :clone(stream[i-1], "weight", "bias", "gradWeight", "gradBias")(stream[i])
    end

    if dropout then
        stream[1] = nn.Dropout(arg.dropout)(stream[1])
    end
    for i=2, arg.numstreams do
        if dropout then
            stream[i] = nn.Dropout(arg.dropout)
                    :clone(stream[i-1], "weight", "bias", "gradWeight", "gradBias")(stream[i])
        end
    end

    return stream
end


local function affine_layer(stream, inputDim, outputDim, dropout)

    stream[1] = nn.Linear(inputDim, outputDim)(stream[1])
    for i=2, arg.numstreams do
        stream[i] = nn.Linear(inputDim, outputDim)
                    :clone(stream[i-1], "weight", "bias", "gradWeight", "gradBias")(stream[i])
    end

    stream[1] = nn.BatchNormalization(outputDim, nil, 0.9)(stream[1])
    for i=2, arg.numstreams do
        stream[i] = nn.BatchNormalization(outputDim, nil, 0.9)
                    :clone(stream[i-1], "weight", "bias", "gradWeight", "gradBias")(stream[i])
    end

    stream[1] = nn.ReLU()(stream[1])
    for i=2, arg.numstreams do
        stream[i] = nn.ReLU()
                    :clone(stream[i-1], "weight", "bias", "gradWeight", "gradBias")(stream[i])
    end

    if dropout then
        stream[1] = nn.Dropout(arg.dropout)(stream[1])
    end
    for i=2, arg.numstreams do
        if dropout then
            stream[i] = nn.Dropout(arg.dropout)
                        :clone(stream[i-1], "weight", "bias", "gradWeight", "gradBias")(stream[i])
        end
    end

    return stream
end


local function affine_out(stream, inputDim, outputDim)
    local out = {}

    out[1] = nn.Linear(inputDim, outputDim)(stream[1])
    out[1] = nn.SoftMax()(out[1])

    for i=2,arg.numstreams do
        out[i] = nn.Linear(inputDim, outputDim)
                    :clone(out[i-1], "weight", "bias", "gradWeight", "gradBias")(stream[i])
        out[i] = nn.SoftMax()
                    :clone(out[i-1], "weight", "bias", "gradWeight", "gradBias")(out[i])
    end

    return out
end


build_model = function(verbose)

    --Input Layer
    local inp = {}
    local h1 = {}
    inp[1] = nn.Identity()()
    for i=2,arg.numstreams do
        inp[i] = nn.Identity()()
    end

    h1[1] = nn.SpatialBatchNormalization(1, nil, 0.9)(inp[1])
    for i=2,arg.numstreams do
        h1[i] = nn.SpatialBatchNormalization(1, nil, 0.9)
                :clone(h1[i-1], "weight", "bias", "gradWeight", "gradBias")(inp[i])
    end

    --Layer1: Conv1 (64, stride=1, 3x3) --Check what exactly was implemented before this!
    h1 = conv_layer(h1, 1, 64)
    -- --Layer2: Conv2 (64, stride=1, 3x3)
    h1 = conv_layer(h1, 64, 64)

    -- --Layer3: Local1
    h1 = local_layer(h1)
    -- --Layer4: Local2
    h1 = local_layer(h1, true)

    -- --Turn into vector, because we only have affine layers after this
    for i=1, arg.numstreams do
        h1[i] = nn.View(-1, 64*8*16)(h1[i])
    end

    -- --Layer5: Affine1 (512 units)
    h1 = affine_layer(h1, 64*16*8, 512, true)
    -- --Layer6: Affine2 (512 units)
    h1 = affine_layer(h1, 512, 512, true)
    -- --Layer7: Affine3 (128 units)
    h1 = affine_layer(h1, 512, 128)

    -- --Output: Affine4 (8 units)
    local output = affine_out(h1, 128, arg.numgestures)
    output = nn.JoinTable(1, arg.numgestures)(output)
    local criterion = nn.CrossEntropyCriterion()

    local g = nn.gModule(inp, {output})
    local parameters, gradParameters = g:getParameters()
	
	if arg.useCuda then
		criterion = criterion:cuda()
		parameters = parameters:cuda()
		gradParameters = gradParameters:cuda()
		g = g:cuda()
	end
    if verbose then
        graph.dot(g.fg, 'fancy', 'fancy')
        print(g)
    end

    return g, criterion, parameters, gradParameters

end

load_model = function()
    local model = th.load(arg.modelFilename)
    local parameters, gradParameters = model:getParameters()
    local criterion = nn.CrossEntropyCriterion()

	if arg.useCuda then
		criterion = criterion:cuda()
		parameters = parameters:cuda()
		gradParameters = gradParameters:cuda()
		model = model:cuda()
	end

    return model, criterion, parameters, gradParameters
end

--[[
Some simple tests:
    - build the model
    - feed-forward
    - feed-backward (backprop)
    - optimise weights
--]]

-- Generate random input
-- local sample_batchsize = 2
-- local sample_X = (th.rand(th.LongStorage{sample_batchsize, 1, 8, 16}))
-- local sample_y = th.FloatTensor({3, 5}):view(-1)

-- local mod, crit, parameters, gradParameters = build_model(true)

-- -- Feed-Forward
-- local logits = mod:forward(sample_X)
-- local loss = crit:forward(logits, sample_y)

-- Feed-Backward (Backprop)

-- Single step of SGD
