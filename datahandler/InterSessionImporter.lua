local th = require "torch"
local lfs = require 'lfs'
--local inspect = require('inspect')
local matio = require 'matio'
local arg = require '../config.lua'
th.setdefaulttensortype('torch.FloatTensor')


local InterSessionImporter = {}
InterSessionImporter.NUM_GESTURES = 8

-- HELPER FUNCTIONS
local function concat(t1, t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

local function shuffle(inp_list)
    math.randomseed(os.time())
    local r = {}
    while #inp_list > 0 do
        table.insert(r, table.remove(inp_list, math.random(#inp_list)))
    end
    return r
end

-- LOCAL FUNCTIONS
local function get_filepaths_in_directory(parent_directory)
    local files = {}
    for entity in lfs.dir(parent_directory) do

        local cur_file_dir = parent_directory .. "/" .. entity

        if entity == "." or entity == ".." or entity[0] == "." then do end
        elseif lfs.attributes(cur_file_dir, "mode") == "file" then
            table.insert(files, cur_file_dir)

        elseif lfs.attributes(cur_file_dir, "mode") == "directory" then
            local tmp = get_filepaths_in_directory(cur_file_dir)
            files = concat(files, tmp)
        end
    end

    if arg.shuffleDevSet then
        files = shuffle(files)
    end

    return files
end


local function get_X_y_sid_from_data(filepaths)
    if #filepaths == 0 then
        print("ERROR 69: No filepaths provided in get_data_from_filepaths")
        os.exit(69)
    end

    for i=1, #filepaths do
        if not filepaths[i] then
            print("Some filepaths seem to be empty strings")
            os.exit(69)
        end
    end

    local sid, data, gesture
    local X_list, y_list, sid_list = {}, {}, {}

    for i=1, #filepaths do

        if arg.dev and (i > arg.devLength) then
            print("dev-set limit reached...")
            break
        end

        data = matio.load(filepaths[i], 'data')
        sid = matio.load(filepaths[i], 'subject')
        gesture = matio.load(filepaths[i], 'gesture')

        data    = th.reshape(data:type('torch.FloatTensor'), 1000, 8, 16)
        sid     = th.repeatTensor(sid:type('torch.FloatTensor'), 1000, 1)
        gesture = th.repeatTensor(gesture:type('torch.FloatTensor'), 1000, 1)

        if filepaths[i] and (not (tonumber(filepaths[i]:sub(56, 58)) == 101 or
            tonumber(filepaths[i]:sub(56, 58)) == 100)) then
            table.insert(X_list, data)
            table.insert(y_list, gesture)
            table.insert(sid_list, sid)
        end
    end

    local X_out = th.cat(X_list, 1)
    local y_out = th.cat(y_list, 1)
    local sid_out = th.cat(sid_list, 1)

    return X_out, y_out, sid_out
end


-- MAIN FUNCTION
function InterSessionImporter.init(data_parent_dir)
    -- Default arguments
    data_parent_dir = data_parent_dir or "../Datasets/Preprocessed/DB-a"

    -- How do we create default parameters?
    local filepaths = get_filepaths_in_directory(data_parent_dir)
    local X, y, sids = get_X_y_sid_from_data(filepaths)

    print("DATA LOADED: InterSessionImporter.lua")
    return X, y, sids

end

print("LOADED: InterSessionImporter.lua")
return InterSessionImporter
