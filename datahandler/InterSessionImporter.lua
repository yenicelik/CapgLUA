local torch = require "torch"
local lfs = require 'lfs'
--local inspect = require('inspect')
local matio = require 'matio'
require '../config.lua'

local InterSessionImporter = {}

InterSessionImporter.NUM_GESTURES = 8
local NUM_GESTURES = 8

torch.setdefaulttensortype('torch.FloatTensor')

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

    local devcounter = 1
    local sid, data, gesture

    local X_list, y_list, sid_list = {}, {}, {}
    local X_out, sid_out, y_out = torch.Tensor(1, 8, 16), torch.Tensor(1, 1), torch.Tensor(1, 1)
    local full = false

    for key, val in pairs(filepaths) do
        devcounter = devcounter + 1

        if (not full) and devcounter > 40 then
            print("dev-set limit reached...")
            break
        end

        data = matio.load(val, 'data'):type('torch.FloatTensor')
        sid = matio.load(val, 'subject'):type('torch.FloatTensor')
        gesture = matio.load(val, 'gesture'):type('torch.FloatTensor')
        if gesture == 101 and NUM_GESTURES == 10 then
            gesture = 10
        elseif gesture == 100 and NUM_GESTURES == 10 then
            gesture = 9
        elseif gesture == 100 or gesture == 101 then do end
        end

        data    = torch.reshape(data, 1000, 16, 8)
        sid     = torch.repeatTensor(sid, 1000, 1)
        gesture = torch.repeatTensor(gesture, 1000, 1)

        X_list[key] = data
        y_list[key] = gesture
        sid_list[key] = sid

    X_out = torch.cat(X_list, 1)
    y_out = torch.cat(y_list, 1)
    sid_out = torch.cat(sid_list, 1)
    end

    return X_out, y_out, sid_out
end


-- MAIN FUNCTION
function InterSessionImporter.init(data_parent_dir, dev)
    -- Default arguments
    data_parent_dir = data_parent_dir or "../Datasets/Preprocessed/DB-a"

    -- How do we create default parameters?
    local filepaths = get_filepaths_in_directory(data_parent_dir)
    local X, y, sids = get_X_y_sid_from_data(filepaths)

    print("... X ... y ... sids ...")
    print(X:size())
    print(y:size())
    print(sids:size())
    print("DATA LOADED: InterSessionImporter.lua")

    return X, y, sids

end

print("LOADED: InterSessionImporter.lua")
return InterSessionImporter
