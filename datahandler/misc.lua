local batchLoader = require "BatchLoader.lua"
require "../config.lua"
-- local X_test = torch.Tensor(1400, 16, 8)
-- local y_test = torch.Tensor(1400)
-- local sids_test = torch.range(1,1401) --torch.Tensor(1400)

-- local test_fnc = {}
-- local gen = torch.Generator()
-- for i=1, 1399 do
-- 	local rnd_num = torch.random(gen, 0, 18)
-- 	local inp_val = torch.Tensor(1):fill(rnd_num)
-- 	-- print("This random number is")
-- 	-- print(inp_val)
-- 	table.insert(test_fnc, inp_val)
-- end

-- local sids_test = torch.cat(test_fnc)

print("Input batchsize is")
print(arg.batchsize)
-- X_batches, y_batches, sid_batches = batchLoader.init(X_test, y_test, sids_test, 50, true)

-- local edone = false
-- while not edone do
-- 	xb, yb, edone = batchLoader.load_batch(X_batches, y_batches, sid_batches)
-- end

local interSessionImporter = require "InterSessionImporter"

X_data, y_data, sid_data = interSessionImporter.init(data_parent_dir)

print("SID DATA IS: ")
print(sid_data:size())
print("X_data")
print(X_data:size())
print("y_data")
print(y_data:size())
print("sid_data")
print(sid_data:size())

X_batches, y_batches, sid_batches = batchLoader.init(X_data, y_data, sid_data, arg.batchsize, true)
--X, y, sids, batch_size, argshuffle)
