-- local batchLoader = require "BatchLoader"

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



-- print("Running batchloader")
-- X_batches, y_batches, sid_batches = batchLoader.init(X_test, y_test, sids_test, 50, true)

-- local edone = false
-- while not edone do
-- 	xb, yb, edone = batchLoader.load_batch(X_batches, y_batches, sid_batches)
-- end

local interSessionImproter = require "InterSessionImporter"

interSessionImproter.init()