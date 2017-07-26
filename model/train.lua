require "../config.lua"
local th = require "torch"
local grad = require "autograd"
local interSessionImporter = require "../datahandler/InterSessionImporter"
local build_model = require "./build_model.lua"
require "optim"

local BatchLoader = require "../datahandler/BatchLoader.lua"

local X_data, y_data, sid_data = interSessionImporter.init()

--New BatchLoader for Train, CV, Test
local batchLoader = BatchLoader:new()

-- Boundaries
local n = X_data:size(1)
local a = math.floor(arg.trainSplit*n)
local b = math.floor(arg.trainSplit*n + arg.cvSplit*n)
print("A, B, N are: ")
print(a)
print(b)
print(n)
print("At train")

local X_train, y_train, sid_train, X_cv, y_cv, sid_cv, X_test, y_test, sid_test = batchLoader:init(
		X_data,
		y_data,
		sid_data,
		10, true
)


--Moving the following function to the top create a bug, where the test-set and cv-set is empty!
local lmodel, lcriterion, lparameters, lgradParameters = build_model(true)

classes = {}
for i=1, tonumber(8) do
	table.insert(classes, i)
end
-- classes = th.LongStorage(classes)

for epoch=1, arg.epochs do
	print("Starting epoch: " .. tostring(epoch))

	local trainConfusion = optim.ConfusionMatrix(classes)

	while not batchLoader.epoch_done do
	    local xBs, yBs = batchLoader:load_batch(X_train, y_train, sid_train)
	    xlua.progress(
	        batchLoader.batch_counter,
	        batchLoader.no_of_batches
	    )

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
		    local loss = lcriterion:forward(logits, targets)

		    -- estimate df/dW
		    local df_dl = lcriterion:backward(logits, targets)
		    lmodel:backward(inputs, df_dl)

		    -- penalties (L1 and L2)
		    local norm = th.norm
		    loss = loss + arg.coefL2 * norm(lparameters, 2)^2/2
		    lgradParameters:add( lparameters:clone():mul(arg.coefL2) )

		    for i = 1, tonumber(arg.batchsize) do
	            trainConfusion:add(logits[i], targets[i])
	         end

		    return loss, lgradParameters
		end

		sgdState = sgdState or {
			learningRate = arg.learningRate,
			momentum = arg.momentum,
			learningRateDevay = 0
		}

		optim.sgd(feval, lparameters, sgdState)

		if batchLoader.batch_counter % tonumber(arg.cvEvery) == 0 then
            print("Cross-Validating...")

			local cvConfusion = optim.ConfusionMatrix(classes)

		    while not batchLoader.epoch_cv_done do
		        local xCV, yCV = batchLoader:load_cvbatch(X_cv, y_cv, sid_cv)

		        local cv_input = xBs[1]:view(-1, 1, 8, 16)
	            local cv_target = yBs[1]:view(-1)

	            arg.testing = true
		        local preds = lmodel:forward(cv_input)
		        arg.testing = false
		        local _, predClass = th.max(preds, 2)
		        for s=1, predClass:size(1) do
		            cvConfusion:add(predClass[s], cv_target[s])
		        end
		    end
		    print(cvConfusion)
		    print("Mean class accuracy (CV) is: " .. tostring(cvConfusion.totalValid * 100))
		    print(trainConfusion)
		    print("Mean class accuracy (Train) is: " .. tostring(trainConfusion.totalValid * 100))
		    batchLoader.epoch_cv_done = false
		end
	end

	batchLoader.epoch_done = false
end

local testConfusion = optim.ConfusionMatrix(classes)

print("Running tests...")
while not batchLoader.epoch_test_done do
	local xTest, yTest = batchLoader:load_test_batch(X_test, y_test, sid_test)

	local test_input = xTest[1]:view(-1, 1, 8, 16)
	local test_target = yTest[1]:view(-1)

	arg.testing = true
	local preds = lmodel:forward(test_input)
	local _, predClass = th.max(preds, 2)
	for s=1, predClass:size(1) do
		testConfusion:add(predClass[s], test_target[s])
	end
end

print(testConfusion)
print("Mean class accuracy (Test) is: " .. tostring(testConfusion.totalValid * 100))

print("Tadaa")