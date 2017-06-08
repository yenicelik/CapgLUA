-- SET PARAMETERS
--pull this into an extract config file which is imported everywhere maybe
print("Config loaded")
arg = lapp[[
    --numgestures       (default 8)                 number of gestures we use
    --dropout           (default 0.5)               passing dropout probability
    --coefL2            (default 0.001)             L2 penalty on the weights
    --batchsize         (default 100)               batch size
    --learningRate      (default 0.01)              learning rate
    --full              (default true)              use the full dataset (not testing model)
    --testing           (default false)             true if recognition/testing phase
    --weightDecay 	    (default 0.001)				L1/L2 weight decay factor
    --momentum		    (default 1)					sgd momentum factor
    --cuda			    (default false)				whether we use cuda or not
]]
