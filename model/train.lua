require "../config.lua"
local th = require "torch"
local grad = require "autograd"
local interSessionImporter = require "../datahandler/InterSessionImporter"
local build_model = require "./build_model.lua"
require "optim"

local BatchLoader = require "../datahandler/BatchLoader.lua"

local X_data, y_data, sid_data = interSessionImporter.init()

--New BatchLoader for Train, CV, Test
local trainLoader = BatchLoader:new()
local cvLoader = BatchLoader:new()
local testLoader = BatchLoader:new()

-- Boundaries
local n = X_data:size(1)
local a = math.floor(arg.trainSplit*n)
local b = math.floor(arg.trainSplit*n + arg.cvSplit*n)
print("A, B, N are: ")
print(a)
print(b)
print(n)
print("At train")

local X_train, y_train, sid_train = trainLoader:init(
	X_data[{{1, a}, {}, {}}],
	y_data[{{1, a}, {}}],
	sid_data[{{1, a}, {}}],
	arg.batchsize, true
)

print("At cv")
local X_cv, y_cv, sid_cv = cvLoader.init(
	cvLoader,
	X_data[{{a, b}, {}, {}}],
	y_data[{{a, b}, {}}],
	sid_data[{{a, b}, {}}],
	arg.batchsize, false
)

print("At test")
local X_test, y_test, sid_test = testLoader:init(
	X_data[{{b, n}, {}, {}}],
	y_data[{{b, n}, {}}],
	sid_data[{{b, n}, {}}],
	arg.batchsize, true
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

	while not trainLoader.epoch_done do
	    local inputs, targets = trainLoader:load_batch(X_train, y_train, sid_train)
	    xlua.progress(
	        trainLoader.batch_counter,
	        trainLoader.no_of_batches
	    )

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

		local sgdState = sgdState or {
			learningRate = arg.learningRate,
			momentum = arg.momentum,
			learningRateDevay = 0
		}

		optim.sgd(feval, lparameters, sgdState)

		if trainLoader.batch_counter % tonumber(arg.cvEvery) == 0 then
            print("Cross-Validating...")

			local cvConfusion = optim.ConfusionMatrix(classes)

		    while not cvLoader.epoch_done do
		        local cv_input, cv_target = cvLoader:load_batch(X_cv, y_cv, sid_cv)

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
		    cvLoader.epoch_done = false
		end
	end

	trainLoader.epoch_done = false
end

local testConfusion = optim.ConfusionMatrix(classes)

print("Running tests...")
while not testLoader.epoch_done do
	local test_input, test_target = testLoader:load_batch(X_test, y_test, sid_test)

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