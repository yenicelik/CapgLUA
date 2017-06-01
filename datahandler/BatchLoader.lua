local BatchLoader = {}

-- BatchLoader.NUM_STREAMS = 3
-- BatchLoader.batch_counter = 1
-- BatchLoader.no_of_batches = 0

NUM_STREAMS = 3
batch_counter = 1 --TODO: Do I need to make this local, such that this is contained in the file only?
no_of_batches = 0
function BatchLoader(X, y, sids, batch_size, argshuffle)
	

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

			--Selecting the first few indecies of the respective queue
			index_of_first_few = categorized_sids[cur_stream][{{1, batch_size}}]:type('torch.LongTensor')			
			--TODO make sure these operators are analgous for higher-sized tensors, this was only tested for sid			
			tmp_x[i] = X:index(1, index_of_first_few:view(index_of_first_few:nElement()))
			tmp_y[i] = y:index(1, index_of_first_few:view(index_of_first_few:nElement())) 
			tmp_sid[i] = sids:index(1, index_of_first_few:view(index_of_first_few:nElement()))

			--Modify or remove dictionary entry
			local len = categorized_sids[cur_stream]:size(1)
			if len - batch_size - 1 < batch_size then
				categorized_sids[cur_stream] = nil
			else
				categorized_sids[cur_stream] = categorized_sids[cur_stream][{{batch_size+1, len}}]	
			end
		end

		if not no_more_full_batches_left then
			no_of_batches = no_of_batches + 1
			X_batches[no_of_batches] = tmp_x
			y_batches[no_of_batches] = tmp_y
			sid_batches[no_of_batches] = tmp_sid
		end
	end

	print(sid_batches)

	return X_batches, y_batches, sid_batches

end


function load_batch(X_batches, y_batches, sid_batches)
	
	local epoch_done = false
	local Xout = X_batches[batch_counter]
	local yout = y_batches[batch_counter]

	batch_counter = batch_counter + 1
	if batch_counter > no_of_batches then
		epoch_done = true
	end

	return Xout, yout, epoch_done

end


function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end



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
X_batches, y_batches, sid_batches = BatchLoader(X_test, y_test, sids_test, 50, true)

local edone = false
while not edone do
	xb, yb, edone = load_batch(X_batches, y_batches, sid_batches)
end
-- qss = torch.Tensor(5):zero()
-- sqq = vector_unique(qss:narrow(1, 2, 3):fill(1))
-- print(sqq)


-- print(test_vec)
-- to_data_dict(test_vec)


-- tmp = vector_unique({1, 2, 3, 4, 2, 4, 1, 4, 5, 2, 1, 5,2 ,2, 5, 9})
-- print(tmp)





