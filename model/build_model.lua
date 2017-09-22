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

--Basic layer functions
local function conv_map_layer(nInputFeatMap, nOutputFeatMap) 
	return nn.SpatialConvolution(nInputFeatMap, nOutputFeatMap, 3,3, 1,1, 1,1)
end

local function local_map_layer()
	return nn.SpatialConvolutionLocal(64, 64, 16, 8, 1,1)
end

local function affine_map_layer(inputDim, outputDim)
	return nn.Linear(inputDim, outputDim)
end

--Normalizers / Special functions
local function spatial_batch_norm_map(nOutputFeatMap)
	return nn.BatchNormalization(nOutputFeatMap, nil, 0.9) --Somehow share specific variables, expect mu and sigma
end

local function batch_norm_map(outputDim)
	return nn.BatchNormalization(outputDim, nil, 0.9) --Somehow share specific variables, expect mu and sigma again 
end

local function dropout_map_layer()
	return nn.Dropout(arg.dropout)
end

--Activation functions
local function relu_map_layer()
	return nn.ReLU()
end

local function softmax_map_layer()
	return nn.SoftMax()
end

--Misc functions
local function id_layer()
	return nn.Identity()
end

local function view_map()
	return nn.View(-1, 64*8*16)
end

--Pflaster functions


build_model = function(verbose)

	local single = nn.Sequential()
	--Layer 1: Conv1 (64, stride=1, 3x3)
	single:add( nn.SpatialConvolution(1, 64, 3,3, 1,1, 1,1) )
	:add( nn.ReLU() )
	--Layer 2: Conv2 (64, stride=1, 3x3)
	:add( nn.SpatialConvolution(64, 64, 3,3, 1,1, 1,1) )
	:add( nn.ReLU() )
	--Transform
	:add( nn.View(-1, 64*8*16) )
	--Layer 5: Affine1 (512 units)
	:add( nn.Linear(8*16*64, 512) )
	:add( nn.Dropout(arg.dropout) )
	--Layer 8: Affine4 (8 units)
	:add( nn.Linear(512, arg.numgestures))

	local parallel = nn.Parallel()
	for i=1, arg.numstreams do
		parallel:add( single )
	end
    if verbose then
		print('Using model: ')
		print(parallel)
	end

	os.exit(0)


	--Final outputs
	local criterion = nil
	if arg.testing then
		single:add( nn.SoftMax() )
	else
		criterion = nn.CrossEntropyCriterion()
	end

    --Core network
	local model = nn.MapTable(single, true)

	local parameters, gradParameters = model:getParameters()

    if verbose then
		print('Using model: ')
		print(model)
	end

	return model, criterion, parameters, gradParameters

end

