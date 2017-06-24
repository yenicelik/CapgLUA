local lfs = require 'lfs'
--local inspect = require('inspect')
local matio = require 'matio'
require '../config.lua'

local InterSessionImporter = {}

InterSessionImporter.NUM_GESTURES = 8


torch.setdefaulttensortype('torch.FloatTensor')

-- HELPER FUNCTIONS
function concat(t1, t2)
	for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
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
	return files
end


local function get_X_y_sid_from_data(filepaths)
	if #filepaths == 0 then
		print("No filepaths provided in get_data_from_filepaths")
		print("Error 69")
		os.exit(69)
	end

	--TODO: it would be much more efficient if we had a list, and we concatenated over that list for each individual X, y and sid (the code running slower with time proves that (I have enough memory I think))
	--TODO: However, I read that lua does not guarantee correct order, and as such, I'm not sure if we can use that

	local out = {}
	local sid, data, gesture
	X_out, sid_out, y_out = torch.Tensor(1, 128), torch.Tensor(1, 1), torch.Tensor(1, 1) --TODO: change this for production
	--X_out = torch.Tensor(1, 16, 8) --change to this for production
	

	X_list, y_list, sid_list = {}, {}, {}

	full = false
	local devcounter = 1
	for key, val in pairs(filepaths) do --so, is it only possible to call a pair-function as an iteration-generator?
		devcounter = devcounter + 1
		if (not full) and devcounter > 100 then
			print("Stopping...")
			break
		end
		gesture = matio.load(val, 'gesture'):type('torch.FloatTensor') --Do I even need to cast this?
		if gesture == 101 and NUM_GESTURES == 10 then
			gesture = 10
		elseif gesture == 100 and NUM_GESTURES == 10 then
			gesture = 9
		elseif gesture == 100 or gesture == 101 then do end
		end
		data = matio.load(val, 'data'):type('torch.FloatTensor')
		--data = torch.reshape(data, torch.LongStorage{1000, 16, 8})
		sid = matio.load(val, 'subject'):type('torch.FloatTensor')

		X_list[key] = data
		y_list[key] = gesture
		sid_list[key] = sid
	
	X_out = torch.cat(X_list, 1) --TODO: does this make sure the order is maintained?
	y_out = torch.cat(y_list)
	sid_out = torch.cat(sid_list)

	end
	return X_out, y_out:t(), sid_out:t()
end


-- MAIN FUNCTION
function InterSessionImporter.init(data_parent_dir, dev)
	-- Default arguments
	local dev = dev or false
	local data_parent_dir = data_parent_dir or "../Datasets/Preprocessed/DB-a"

	-- How do we create default parameters?
	local filepaths = get_filepaths_in_directory(data_parent_dir)
	local X, y, sids = get_X_y_sid_from_data(filepaths)
	return X, y, sids

end

return InterSessionImporter
-- local filepaths = get_filepaths_in_directory("../Datasets/Preprocessed/DB-a")
-- get_data_from_filepaths(filepaths)


