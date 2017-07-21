require "../config.lua"
local interSessionImporter = require "InterSessionImporter"
local batchLoader = require "BatchLoader.lua"

X_data, y_data, sid_data = interSessionImporter.init(data_parent_dir)

X_batches, y_batches, sid_batches = batchLoader.init(X_data, y_data, sid_data, arg.batchsize, true)
--X, y, sids, batch_size, argshuffle)

-- X_batches, y_batches, sid_batches = batchLoader.init(X_test, y_test, sids_test, arg.batchsize, true)

-- local edone = false
-- while not edone do
-- 	xb, yb, edone = batchLoader.load_batch(X_batches, y_batches, sid_batches)
-- end
