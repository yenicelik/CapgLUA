require "../config.lua"
local interSessionImporter = require "InterSessionImporter"
local batchLoader = require "BatchLoader.lua"

X_data, y_data, sid_data = interSessionImporter.init(data_parent_dir)

X_batches, y_batches, sid_batches = batchLoader.init(X_data, y_data, sid_data, arg.batchsize, true)

while not batchLoader.epoch_done do
	xb, yb = batchLoader.load_batch(X_batches, y_batches, sid_batches)
end
