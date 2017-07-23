-- SET PARAMETERS
--pull this into an extract config file which is imported everywhere maybe
-- BEWARE THIS IS RE-CREATED ON EACH CALL!
--Booleans: true is interpreted as a boolean; false is interpreted as a string (if put into lapp above)
arg = nil
if not arg then
    arg = {
        numgestures     = 8,                 --number of gestures we use
        dropout         = 0.5,               --passing dropout probability
        coefL2          = 0.001,             --L2 penalty on the weights
        batchsize       = 10,                --batch size
        learningRate    = 0.01,              --learning rate
        weightDecay     = 0.001,             --L1/L2 weight decay factor
        momentum        = 1,                 --sgd momentum factor
        devLength       = 30,               --how many files should be read for dev-mode
        --For parameter 'dev': false gives me a string, true gives me a boolean...
        dev             = true,              --use the full dataset (not testing model)
        testing         = false,             --true if recognition/testing phase
        cuda            = false,             --whether we use cuda or not
        shuffleDevSet   = true,              --shuffle the development set
        trainSplit      = 0.5,               --percentage of data to use for training
        cvSplit         = 0.2,               --percentage of data to use for cross-validation
        testSplit       = 0.3,               --percentage of data to use for testing (final model evaluation)
        cvEvery         = 50,                 --how often to cross-validate
        epochs          = 2,                --total number of epochs to run

    }
end

--trainSplit, cvSplit and testSplit should sum to 1
if arg.trainSplit + arg.cvSplit +arg.testSplit ~= 1.0 then
    print("config.lua: Not all the data is used, or more than the entire dataset is used!")
    os.exit(69)
end


print("LOADED: config.lua loaded")
return arg

