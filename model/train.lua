require "../config.lua"
local th = require "torch"
local grad = require "autograd"
local interSessionImporter = require "../datahandler/InterSessionImporter"
local batchLoader = require "../datahandler/BatchLoader.lua"
local build_model = require "./build_model.lua"

local X_data, y_data, sid_data = interSessionImporter.init()
local X_batches, y_batches, sid_batches = batchLoader.init(X_data, y_data, sid_data, arg.batchsize, true)
local lmodel, lcriterion, lparameters, lgradParameters = build_model(true)


while not batchLoader.epoch_done do
    local xBs, yBs = batchLoader.load_batch(X_batches, y_batches, sid_batches)
    xlua.progress(batchLoader.batch_counter, batchLoader.no_of_batches)

    local inputs = xBs[1]:view(-1, 1, 8, 16)
    local targets = yBs[1]:view(-1)

	local feval = function(x)
	    collectgarbage()

	    if x ~= lparameters then
	        lparameters:copy(x)
	    end

	    --reset gradients
	    lgradParameters:zero()

	    --evaluate function for complete minibatch
	    local logits = lmodel:forward(inputs)

        print("Targets: ")
        print(targets)
        print("Logits: ")
        print(logits)

	    local loss = lcriterion:forward(logits, targets)

	    -- estimate df/dW
	    local df_dl = lcriterion:backward(logits, targets)
	    lmodel:backward(inputs, df_dl)

	    -- penalties (L1 and L2)
	    local norm = th.norm
	    loss = loss + arg.coefL2 * norm(lparameters, 2)^2/2
	    lgradParameters:add( lparameters:clone():mul(arg.coefL2) )

	    return loss, lgradParameters
	end

	sgdState = sgdState or {
		learningRate = arg.learningRate,
		momentum = arg.momentum,
		learningRateDevay = 0
	}

	optim.sgd(feval, lparameters, sgdState)

end
