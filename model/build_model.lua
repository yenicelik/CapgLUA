local th = require('torch')
local nn = require 'nn'
require 'pl'
require '../config.lua'
require '../main.lua'

th.setdefaulttensortype('torch.FloatTensor')
--[[
    TODO:
    - Check if batchnorm parameters are correct
--]]

-- [[LAYERS]]
local function conv_layer(model, nInputFeatMap, nOutputFeatMap)
    model:add(nn.SpatialConvolution(nInputFeatMap, nOutputFeatMap, 3,3, 1,1, 1,1))
        :add(nn.SpatialBatchNormalization(nOutputFeatMap))
        :add(nn.ReLU())
end

local function local_layer(model, dropout)
    model:add(nn.SpatialConvolutionLocal(64, 64, 16, 8, 1,1)) --Swapping 16 and 8 causes an error.. why?
        :add(nn.SpatialBatchNormalization(64))
        :add(nn.ReLU())
    if dropout then
        model:add(nn.Dropout(arg.dropout))
    end
end

local function affine_layer(model, inputDim, outputDim, dropout)
    model:add(nn.Linear(inputDim, outputDim))
        :add(nn.BatchNormalization(outputDim, nil, 0.9))
        :add(nn.ReLU())
    if dropout then
        model:add(nn.Dropout(arg.dropout))
    end
end


-- MODEL INITIALIZER
build_model = function(verbose)
    local model = nn.Sequential()

    --Input Layer
    model:add( nn.SpatialBatchNormalization(1) )

    --Layer1: Conv1 (64, stride=1, 3x3)
    conv_layer(model, 1, 64) --Check what exactly was implemented before this!

    --Layer2: Conv2 (64, stride=1, 3x3)
    conv_layer(model, 64, 64)

    --Layer3: Local1
    local_layer(model)

    --Layer4: Local2
    local_layer(model, true)

    --Turn into vector, because we only have affine layers after this
    model:add(nn.View(-1, 64*8*16))

    --Layer5: Affine1 (512 units)
    affine_layer(model, 8*16*64, 512, true)

    --Layer6: Affine2 (512 units)
    affine_layer(model, 512, 512, false)

    --Layer7: Affine3 (128 units)
    affine_layer(model, 512, 128, false)

    --Output: Affine4 (8 units)
    model:add(nn.Linear(128, arg.numgestures, true))

    --Such that we don't apply a softmax operation twice
    local criterion = nil
    if arg.testing then
        model:add(nn.SoftMax())
    else
        --Any operation needed, and does this effectively replace the softmax option?
        criterion = nn.CrossEntropyCriterion()
    end

    local parameters, gradParameters = model:getParameters()

    if verbose then
        print('Using model: ')
        print(model)
    end

    return model, criterion, parameters, gradParameters

end

return build_model

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
