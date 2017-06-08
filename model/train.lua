require 'optim'
require 'build_model'
require 'xlua'


classes = torch.range(0, arg.numgestures)
model, criterion = build_model()

parameters, parametersGradients = model:getParameters()
print(model)

-- CUDA -----------------------------------------------------------------------
if arg.cuda == 'cuda' then
   model:cuda()
   criterion:cuda()
end

-- DEFINING THE OPTIMIZER -----------------------------------------------------
optimState = {
      learningRate = arg.learningRate,
      weightDecay = arg.weightDecay,
      momentum = arg.momentum,
      learningRateDecay = 0 	--Need to just change this at some point
   }
optimMethod = optim.sgd

-- DEFINING TRAINING PROCEDURE ------------------------------------------------
function train()
	epoch = epoch or 1

	print(arg)

	model:training()
   	print("==> epoch # " .. epoch .. ' [batchSize = ' .. arg.batchsize .. ']')

   	--Doing one full epoch
   	--Per batch do this
   	input = torch.rand(arg.batchsize, 1, 8, 16)
   	output = torch.rand(arg.batchsize, 2)
   	print(input:size())
   	print(output:size())
	
	-- sample_batchsize = 2
	-- sample_X = torch.rand(sample_batchsize, 1, 8, 16)
	-- sample_y = torch.rand(sample_batchsize, arg.numgestures)
   	
   	-- feed it to the neural network and the criterion
	criterion:forward(model:forward(input), output)

	-- train over this example in 3 steps
	-- (1) zero the accumulation of the gradients
	model:zeroGradParameters()
	-- (2) accumulate gradients
	model:backward(input, criterion:backward(model.output, output))
	-- (3) update parameters with a 0.01 learning rate
	model:updateParameters(0.01)

end

for epoch = 1, 20 do
	train()
end


-- require 'optim'

-- local opt = lapp[[
--    -s,--save          (default "logs")      subdirectory to save logs
--    -n,--network       (default "")          reload pretrained network
--    -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
--    -f,--full                                use the full dataset
--    -p,--plot                                plot while training
--    -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
--    -r,--learningRate  (default 0.05)        learning rate, for SGD only
--    -b,--batchSize     (default 10)          batch size
--    -m,--momentum      (default 0)           momentum, for SGD only
--    -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
--    --coefL1           (default 0)           L1 penalty on the weights
--    --coefL2           (default 0)           L2 penalty on the weights
--    -t,--threads       (default 4)           number of threads
--    -g,--numgestures   (default 8)           number of gesture we use in this training
--    -d,--dropout       (default 0.5)         dropout probability
-- ]]

-- -- Is the thready property any useful here?
-- if opt.optimization == 'SGD' then
--     torch.setdefaulttensortype('torch.FloatTensor')
-- end

-- classes = torch.range(0, opt.numgestures)
-- geometry = {8,16}
-- p = 0.5


-- ----- COPY THIS INTO A SEPARATE CONFIG FILE

-- confusion = optim.ConfusionMatrix(classes) --must include classes from prior app

-- -- Loggers 
-- trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
-- testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))


-- function run_on_batch(X, y, epoch)
-- 	epoch = epoch or 1

-- 	local time = sys.clock()
-- 	--Do one epoch: while done do end

-- 	--Closure to evaluate f(X) and df/dX
-- 	local feval = function(x)
-- 		collectgarbage()

-- 		--get new parameters
-- 		if x ~= parameters then
-- 			parameters:copy(x)
-- 		end

-- 		--reset gradients
-- 		gradParameters:zero()

-- 		--evaluate function for mini-batch
-- 		local outputs = model:forward(X)
-- 		local f = criterion:forward(outputs, targets) --actual loss

-- 		--estimate df/dW
-- 		local df_do = criterion:backward(outputs, target)
-- 		model:backward(X, df_do)

-- 		--update confusion
-- 		for i=1, opt.batchsize do
-- 			confusion:add(outputs[i], targets[i])
-- 		end

-- 		return f, gradParameters
-- 	end


-- 	--Perform SGD
-- 	sgdState = sgdState or {
-- 		learningRate = opt.learningRate,
-- 		momentum = opt.momentum,
-- 		learningRateDecay = 5e-7
-- 	}

-- 	optim.sgd(feval, parameters, sgdState)

-- end



-- function test_on_batch(X, y)
-- 	for i=1, number_of_batches do 

-- 		predictions = model:forward(X)

-- 		for j=1, #classes do
-- 			confusion:add(predictions[j], y[j])
-- 		end

-- 	end
-- end







