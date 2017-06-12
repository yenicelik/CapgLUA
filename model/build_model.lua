-- INITIAL OPERATIONS AND IMPORTS
require('torch')
require 'nn'
require 'pl'
require '../config'
require '../main'

torch.setdefaulttensortype('torch.FloatTensor')

-- DEFINING THE MODEL
local model = nn.Sequential()

-- LAYERS
local function conv_layer(nInputFeatMap, nOutputFeatMap)
    model:add(nn.SpatialConvolution(nInputFeatMap, nOutputFeatMap, 3,3, 1,1, 1,1))
        :add(nn.SpatialBatchNormalization(nOutputFeatMap))
        :add(nn.ReLU())
end

local function local_layer(dropout)
        --[[
    Das dritte und vierte Argument für den locally-connected conv layer sind 
    die input width/height. Für das Modell denn du unten definiert hast, ist
    das 8x16]]
    model:add(nn.SpatialConvolutionLocal(64, 64, 16,8, 1,1)) --somehow crashed width and height are changed...
        :add(nn.SpatialBatchNormalization(64))
        :add(nn.ReLU())
    if dropout then
        model:add(nn.Dropout(arg.dropout))
    end
end

local function affine_layer(inputDim, outputDim, dropout)
    model:add(nn.Linear(inputDim, outputDim))
        :add(nn.BatchNormalization(outputDim, nil, 0.9))
        :add(nn.ReLU())
    if dropout then
        model:add(nn.Dropout(arg.dropout))
    end
end

-- MODEL INITIALIZER
function build_model()
    --Input: Input layer
    --[[
    Verstehe nicht ganz, warum du das brauchst.
    ]]
    -- model:add( nn.View(16*8, 1) )
    -- model:add( nn.BatchNormalization(16*8, nil, 0.9) )

    --Layer1: Conv1 (64, stride=1, 3x3)
    conv_layer(1, 64) --Check what exactly was implemented before this!
    --Layer2: Conv2 (64, stride=1, 3x3)
    conv_layer(64, 64)

    --Layer3: Local1 
    local_layer()
    --Layer4: Local2 
    local_layer(true)

    model:add(nn.View(-1, 64*8*16))

    --Layer5: Affine1 (512 units)
    affine_layer(8*16*64, 512, true)
    --Layer6: Affine2 (512 units)
    affine_layer(512, 512, false)
    --Layer7: Affine3 (128 units)
    affine_layer(512, 128, false)

    --Output: Affine3 (8 units)
    model:add(nn.Linear(128, arg.numgestures, true))

    --Such that we don't apply a softmax operation twice 
    if arg.training then
        model:add(nn.SoftMax())
    else
        criterion = nn.CrossEntropyCriterion()
    end

    return model, criterion

end


-- parameters, gradParameters = model:getParameters()


-- BUILD THE MODEL
-- build_model()
-- arg.training = false
-- output_model()
-- print("Using model")
-- print(model)

--------------------------------------
-- loss function: Softmax Cross Entropy
-- does this apply softmax twice with the output layer?
--criterion = nn.CrossEntropyCriterion() --TODO how do i make sure the inputs here are turned into 'log's first

sample_batchsize = 2
sample_X = torch.rand(sample_batchsize, 1, 8, 16)
sample_y = torch.rand(sample_batchsize, arg.numgestures)

-- logits = model:forward(sample_X)


-- sample_batchsize = 1
-- sample_X = torch.rand(sample_batchsize, imgSize[1], imgSize[2])
-- sample_y = torch.rand(sample_batchsize, opt.numgestures)

print(sample_X:size())
print(sample_y:size())

logits = model:forward(sample_X)
-- loss = criterion:forward(sample_y, logits)














