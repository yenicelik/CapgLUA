local BatchLoader = {}

BatchLoader.NUM_STREAMS = 3
BatchLoader.batch_counter = 1 --TODO: Do I need to make this local, such that this is contained in the file only?
BatchLoader.no_of_batches = 0

-- HELPER FUNCTIONS
local function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

-- INITIALIZER
function BatchLoader.init(X, y, sids, batch_size, argshuffle)
	
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

		for i=1, BatchLoader.NUM_STREAMS do 

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
			BatchLoader.no_of_batches = BatchLoader.no_of_batches + 1
			X_batches[BatchLoader.no_of_batches] = tmp_x
			y_batches[BatchLoader.no_of_batches] = tmp_y
			sid_batches[BatchLoader.no_of_batches] = tmp_sid
		end
	end

	print(sid_batches)

	return X_batches, y_batches, sid_batches

end

-- LOADER FUNCTION
function BatchLoader.load_batch(X_batches, y_batches, sid_batches)
	
	local epoch_done = false
	local Xout = X_batches[batch_counter]
	local yout = y_batches[batch_counter]

	BatchLoader.batch_counter = BatchLoader.batch_counter + 1
	if BatchLoader.batch_counter > BatchLoader.no_of_batches then
		epoch_done = true
	end

	return Xout, yout, epoch_done

end




return BatchLoader





