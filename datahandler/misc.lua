require "../config.lua"
local interSessionImporter = require "InterSessionImporter"
local BatchLoader = require "BatchLoader.lua"

local X_data, y_data, sid_data = interSessionImporter.init()

local batchLoader = BatchLoader:new()
local X_batches, y_batches, sid_batches = batchLoader:init(X_data, y_data, sid_data, arg.batchsize, true)

while not batchLoader.epoch_done do

    local xb, yb = batchLoader:load_batch(X_batches, y_batches, sid_batches)
end
