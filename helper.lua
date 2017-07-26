hf = {}

function hf.take_n(n, inp_table)
    local out = {}
    if n == 0 then
        print("ERROR: n is 0 in take_n!")
        os.exit(69)
    end
    if n > #inp_table then
        print("ERROR: n is bigger than the table!")
        os.exit(69)
    end
    while #out < n do
        table.insert(out, table.remove(inp_table, 1))
    end
    return out
end

-- -- Some test cases to visuall check:
if false then
    local t1 = hf.take_n(5, {'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', })
    if t1 ~= {'x', 'x', 'x', 'x', 'x'} then
        print("Actual: ")
        print(t1)
        print("Expected: ")
        print({'x', 'x', 'x', 'x', 'x'})
    end

    local t2 = hf.take_n(5, {52, 15, 83, 42, 94, 20, 55, 85})
    if t2 ~= {52, 15, 83, 42, 94} then
        print("Actual: ")
        print(t2)
        print("Expected: ")
        print({52, 15, 83, 42, 94})
    end
    os.exit(0)
end

return hf