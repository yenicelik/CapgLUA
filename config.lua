-- SET PARAMETERS
--pull this into an extract config file which is imported everywhere maybe
-- BEWARE THIS IS RE-CREATED ON EACH CALL!
--Booleans: true is interpreted as a boolean; false is interpreted as a string (if put into lapp above)
arg = nil
if not arg then
    arg = {
        numgestures     = 8,                 --number of gestures we use
        devLength       = 100,               --how many files should be read for dev-mode
        dropout         = 0.5,               --passing dropout probability
        coefL2          = 0.001,             --L2 penalty on the weights
        batchsize       = 10,                --batch size
        learningRate    = 0.01,              --learning rate
        weightDecay     = 0.001,             --L1/L2 weight decay factor
        momentum        = 1,                 --sgd momentum factor
        dev             = false,              --use the full dataset (not testing model) / false gives me a string, true gives me a boolean...
        testing         = false,             --true if recognition/testing phase
        cuda            = false,             --whether we use cuda or not
        shuffleDevSet   = true              --shuffle the development set
    }
end



print("LOADED: config.lua loaded")
return arg

