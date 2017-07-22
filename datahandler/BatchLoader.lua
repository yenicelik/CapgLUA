local th = require "torch"
require "../config.lua"
local hf = require "../helper.lua"

local BatchLoader = {}
-- Stuff
BatchLoader.NUM_STREAMS = 8 --Outsource this
BatchLoader.batch_counter = 1 --TODO: Do I need to make this local, such that this is contained in the file only?
BatchLoader.no_of_batches = 0
BatchLoader.epoch_done = false

-- INITIALIZER
function BatchLoader.init(X, y, sids, batch_size, argshuffle)
    --Shuffling items
    if argshuffle then
        local perm = th.randperm(sids:size(1)):long()
        X = X:index(1, perm)
        y = y:index(1, perm)
        sids = sids:index(1, perm)
    end

    --Creating table of included sub-indecies
    local session_ids = {}
    for i=0, 18 do
        session_ids[i] = {}
    end

    for j=1, sids:view(-1):size(1) do
        table.insert(session_ids[sids:view(-1)[{j}]], j)
    end

    --Generating batches
    local X_batches = {}
    local y_batches = {}
    local sid_batches = {}
    local no_more_full_batches_left = false
    local used_samples = 0

    while not no_more_full_batches_left do
        local tmp_x = {}
        local tmp_y = {}
        local tmp_sid = {}

        for i=1, BatchLoader.NUM_STREAMS do
            --Choose random key from dictionary
            local possible_indecies = {}
            for key, _ in pairs(session_ids) do
                if next(session_ids[key]) then
                    table.insert(possible_indecies, key)
                end
            end

            --Delete key if not enough (batch_size) samples existent to fill out all three parallel batches
            if  sids:view(-1):size(1) - used_samples < batch_size then
                print("BatchLoader: Max number of session_ids used")
                no_more_full_batches_left = true
                break
            end

            math.randomseed(os.time())
            local cur_stream = possible_indecies[math.random(#possible_indecies)]
            local index_of_first_few = hf.take_n(batch_size, session_ids[cur_stream])

            tmp_x[i] = X:index(1, th.LongTensor(index_of_first_few))
            tmp_y[i] = y:index(1, th.LongTensor(index_of_first_few))
            tmp_sid[i] = sids:index(1, th.LongTensor(index_of_first_few))

            local sum = 0
            for j=1, #session_ids do
                sum = sum + #session_ids[j]
            end

            used_samples = used_samples + batch_size
        end

        -- Putting the newly generated 'parallel' strem into a batch.
        -- BatchLoader.no_of_batches increases with every new batch we put in
        if not no_more_full_batches_left then
            BatchLoader.no_of_batches = BatchLoader.no_of_batches + 1
            X_batches[BatchLoader.no_of_batches] = tmp_x
            y_batches[BatchLoader.no_of_batches] = tmp_y
            sid_batches[BatchLoader.no_of_batches] = tmp_sid
        end
    end

    -- The output has the form [(Batch1), (Batch2)]
    -- where Batch_i has the form (input_for_stream_1, .., input_for_stream_i, ..)
    print("CREATED: All batches generated")
    return X_batches, y_batches, sid_batches

end

-- LOADER FUNCTION
function BatchLoader.load_batch(X_batches, y_batches)

    local Xout = X_batches[BatchLoader.batch_counter]
    local yout = y_batches[BatchLoader.batch_counter]

    BatchLoader.batch_counter = BatchLoader.batch_counter + 1
    if BatchLoader.batch_counter > BatchLoader.no_of_batches then
        BatchLoader.epoch_done = true
        BatchLoader.batch_counter = 1
        print("EPOCH DONE: Batchloader.lua")
    end

    return Xout, yout
end

print("LOADED: BatchLoader.lua")

return BatchLoader
