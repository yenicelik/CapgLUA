require('torch')
require 'nn'

NUM_GESTURES = 8
p = 0.5
model = nn.Sequential()

local function print_module()
    for i,module in ipairs(model:listModules()) do
       print(module)
       print(module.output:size())
    end
end

local function conv_layer()
    model:add( nn.View(16, 8) )
    model:add( nn.SpatialConvolution(1, 64, 3,3, 1,1, 1,1))
    model:add( nn.View(16*8) ) --TODO do we need the '1' over here?
    model:add( nn.BatchNormalization(16*8, nil, 0.9, nil) )
    model:add( nn.ReLU() )
    return model --Is this valid like this?
end

local function local_layer(dropout)
    dropout = dropout or false
    model:add( nn.View(16, 8) )
    model:add( nn.SpatialConvolutionLocal(64, 64, 3,3, 1,1) )
    model:add( nn.View(16*8) ) --TODO do we need the '1' over here?
    model:add( nn.BatchNormalization(16*8, nil, 0.9, nil) )
    model:add( nn.ReLU() )
    if dropout then
        model:add( nn.Dropout(p) )
    end
    return model
end

local function affine_layer(inputDim, outputDim, dropout)
    dropout = dropout or false
    print("inputDim is")
    print(inputDim)
    print("outpuDim is")
    print(outputDim)
    model:add( nn.Linear(inputsDim, outputDim, true) )
    model:add( nn.BatchNormalization(outputDim, nil, 0.9, nil) )
    model:add( nn.ReLU() )
    if dropout then
        model:add( nn.Dropout(p) )
    end
    return model
end

function build_model()
    
    --Input: Input layer
    model:add( nn.View(16*8, 1) ) --TODO do we need the '1' over here?
    model:add( nn.BatchNormalization(16*8, nil, 0.9, nil) ) -- You can also define: eps, affine

    --Layer1: Conv1 (64, stride=1, 3x3)
    conv_layer()
    --Layer2: Conv2 (64, stride=1, 3x3)
    conv_layer()

    --Layer3: Local1 
    local_layer(false)
    --Layer4: Local2 
    local_layer(true)

    --Layer5: Affine1 (512 units)
    affine_layer(16*8*64, 512, true)
    --Layer6: Affine2 (512 units)
    affine_layer(512, 512, false)
    --Layer7: Affine3 (128 units)
    affine_layer(512, 128, false)

    --Output: Affine3 (8 units)
    model:add( nn.Linear(128, NUM_GESTURES, true) )
    model:add( SoftMax() )

    print_module()

    return model
    
end

build_model()
