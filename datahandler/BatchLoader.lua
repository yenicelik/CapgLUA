NUM_STREAMS = 3
batch_counter = 0 --TODO: do I need to add a local here?
samples = 0

function BatchLoader(X, y, sids, batch_size, argshuffle)
	no_of_batches = 1

	--Shuffling items
	if argshuffle then 
		local perm = torch.randperm(sids:size(1)):long()
		X = X:index(1, perm) --TODO make sure this actually shuffles the data!
		y = y:index(1, perm)
		sids = sids:index(1,perm)
	end

	--Creating table of included sub-indecies
	local categorized_sids = {}
	for i=0, 18 do
		categorized_sids[i] = sids:eq(i):nonzero()
		if categorized_sids[i]:size(1) < batch_size then
			categorized_sids[i] = nil
		end
	end

	--Generating batches
	local X_batches = {}
	local y_batches = {}
	local sid_batches = {}
	local no_more_full_batches_left = false

	while not no_more_full_batches_left do
		
		local tmp_x = {}
		local tmp_y = {}
		local tmp_sid = {}
		local cur_random_stream

		for i=1, NUM_STREAMS do 

			--TODO check if enough batches exist
			--Delete key if not enough (batch_size) samples existent
			if tablelength(categorized_sids) == 0 then
				no_more_full_batches_left = true
				break
			end
			
			--Choose random key from dictionary
			local all_indecies = {}
			for key, value in pairs(categorized_sids) do
				table.insert(all_indecies, key)
			end
			gen = torch.Generator()
			cur_stream = all_indecies[torch.random(gen, 1, #all_indecies)]

			--This should select the first batch_size items
			--Save first (batch_size) samples into the temporary variables
			first_batch_item_indecies = categorized_sids[cur_stream][{{1, batch_size}}]			
			tmp_x[i] = X[{{1, first_batch_item_indecies}, {}, {}}]
			tmp_y[i] = y[{{1, first_batch_item_indecies}}]
			tmp_sid[i] = sids[{first_batch_item_indecies}]

			print(tmp_sid[i])

			-- print("This should grow")
			-- print(tmp_sid)

			--Modify or delete the queue of the respective cur_stream
			local len = categorized_sids[cur_stream]:size(1)
			if len - batch_size - 1 < batch_size then
				categorized_sids[cur_stream] = nil
			else
				categorized_sids[cur_stream] = categorized_sids[cur_stream][{{batch_size+1, len}}]	
			end
		end

		if not no_more_full_batches_left then
			X_batches[no_of_batches] = tmp_x
			y_batches[no_of_batches] = tmp_y
			sid_batches[no_of_batches] = tmp_sid
			no_of_batches = no_of_batches + 1
		end
	end

	print(sid_batches[1][1])
	print(type(sid_batches[1][1]))

	return X_batches, y_batches, sid_batches

end


function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

-- local function to_data_dict(argsids)
-- 	local out = {};

-- 	for i=0, 18 do
-- 		out[i] = torch.Tensor(1)
-- 	end

-- 	print(out)

-- 	for i=1, argsids:size(1) do --TODO does
-- 		print(argsids[i])
-- 		print(out[argsids[i]])
-- 		out[argsids[i]]:insert(i)
-- 	end
-- 	return out
-- end


X_test = torch.Tensor(1400, 16, 8)
y_test = torch.Tensor(1400)
sids_test = torch.range(1,1401) --torch.Tensor(1400)

local test_fnc = {}
gen = torch.Generator()
for i=1, 1399 do
	local rnd_num = torch.random(gen, 0, 18)
	local inp_val = torch.Tensor(1):fill(rnd_num)
	-- print("This random number is")
	-- print(inp_val)
	table.insert(test_fnc, inp_val)
end

sids_test = torch.cat(test_fnc)



print("Running batchloader")
BatchLoader(X_test, y_test, sids_test, 10, true)
-- qss = torch.Tensor(5):zero()
-- sqq = vector_unique(qss:narrow(1, 2, 3):fill(1))
-- print(sqq)


-- print(test_vec)
-- to_data_dict(test_vec)


-- tmp = vector_unique({1, 2, 3, 4, 2, 4, 1, 4, 5, 2, 1, 5,2 ,2, 5, 9})
-- print(tmp)





