require('torch')
require 'nn'

NUM_GESTURES = 8
p = 0.5
model = nn.Sequential()

--[[
Du kannst einfach print(model) aufrufen, das listed dein ganzes Model
]] 
-- local function print_module()
--     for i,module in ipairs(model:listModules()) do
--        print(module)
--        print(module.output:size())
--     end
-- end

--[[
Ein kleiner 'Hack' denn du anwenden kannst bei Modeldefinitionen, ist 
folgender. Anstelle von:

model:add(nn.Linear(10,10))
model:add(nn.Linear(10,20))
model:add(...)
...

Mache:

model:add(nn.Linear(10,10))
    :add(nn.Linear(10,20))
    :add(...)
...

Es erspart dir schreibarbeit und sieh übersichtlicher aus. Das kann man machen,
weil add() das Modell wieder zurückgibt. Ich werde es in deinen Code anwenden.
]]

--[[
Diese Funktion braucht Parameters. Siehe unten in 'build_model'
]]
function conv_layer(nInputFeatMap, nOutputFeatMap)
    --[[
    nn.View brauchst du hier nicht. Die Daten sollten laut Paper 1x8x16 sein.
    ]]
    -- model:add(nn.View(16, 8))
    model:add(nn.SpatialConvolution(nInputFeatMap, nOutputFeatMap, 3,3, 1,1, 1,1))
    --[[
    Es gibt die SpatialBatchNormalization funktion. Du musst nicht die View 
    ändern
    ]]
    -- model:add( nn.View(16*8) ) --TODO do we need the '1' over here?
    -- model:add( nn.BatchNormalization(16*8, nil, 0.9, nil) )
        :add(nn.SpatialBatchNormalization(nOutputFeatMap))
        :add(nn.ReLU())
    --[[
    Du manipulierst das Model direkt, musst es also nicht zurückgeben
    ]]
    -- return model --Is this valid like this?
end

function local_layer(dropout)
    --[[
    nil wird als false ausgewertet in Lua. Dann musst du keinen Bool Parameter
    eingeben wenn du kein dropout haben möchtest
    ]]
    --dropout = dropout or false
    -- model:add(nn.View(16, 8))
    --[[
    Das dritte und vierte Argument für den locally-connected conv layer sind 
    die input width/height. Für das Modell denn du unten definiert hast, ist
    das 8x16]]
    model:add(nn.SpatialConvolutionLocal(64, 64, 16,8, 1,1))
    -- model:add( nn.View(16*8) ) --TODO do we need the '1' over here?
    -- model:add( nn.BatchNormalization(16*8, nil, 0.9, nil) )
        :add(nn.SpatialBatchNormalization(64))
        :add(nn.ReLU())
    if dropout then
        model:add(nn.Dropout(p))
    end
    -- return model
end

function affine_layer(inputDim, outputDim, dropout)
    --dropout = dropout or false
    print("inputDim is")
    print(inputDim)
    print("outpuDim is")
    print(outputDim)

    --[[
    Bias = true ist default bein nn.Linear
    ]]
    model:add(nn.Linear(inputDim, outputDim))
    --[[
    Letzten param brauchst du nicht als nil einzugeben. Wenn dus weglässt, ist
    es bei default nil.
    ]]
        :add(nn.BatchNormalization(outputDim, nil, 0.9))
        :add(nn.ReLU())
    if dropout then
        model:add(nn.Dropout(p))
    end
    -- return model
end

function build_model()
    
    --Input: Input layer
    --[[
    Verstehe nicht ganz, warum du das brauchst.
    ]]
    -- model:add( nn.View(16*8, 1) ) --TODO do we need the '1' over here?
    -- model:add( nn.BatchNormalization(16*8, nil, 0.9) ) -- You can also define: eps, affine


    --[[
    Das kannst du so nicht machen. Die input/output dim müssen übereinstimmen.
    So wie du das geschrieben, würde dies mit deiner alten Funktion folgende
    layers bauen:
    conv_layer(1, 64)
    ...
    conv_layer(1, 64)
    ...
    Jedoch sollte es wie folgt sein:
    conv_layer(1, 64)
    ...
    conv_layer(64, 64)
    Ich habe die Funktion geändert damit das klappt.]]
    -- --Layer1: Conv1 (64, stride=1, 3x3)
    -- conv_layer()
    -- --Layer2: Conv2 (64, stride=1, 3x3)
    -- conv_layer()
    conv_layer(1, 64)
    conv_layer(64, 64)

    --Layer3: Local1 
    local_layer()
    --Layer4: Local2 
    local_layer(true)

    --[[
    Hier brauchst du nn.View, da nn.Linear einen Vektor und keine Matrix 
    akzeptiert. -1 gibt an, dass du diesen Parameter von den Restlichen 
    herleiten willst (i.e die Batch size). Du kannst auch nn.Reshape verwenden,
    ist jedoch ein wenig ineffizienter, da es einen Memorycopy macht. Brauchst
    dafür aber den -1 Parameter nicht.]]
    model:add(nn.View(-1, 64 * 8 * 16))

    --Layer5: Affine1 (512 units)
    affine_layer(16*8*64, 512, true)
    --Layer6: Affine2 (512 units)
    affine_layer(512, 512, false)
    --Layer7: Affine3 (128 units)
    affine_layer(512, 128, false)

    --Output: Affine3 (8 units)
    model:add(nn.Linear(128, NUM_GESTURES, true))
    model:add(nn.SoftMax())

    print(model)

    return model
    
end

build_model()
