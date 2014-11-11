

require 'torch'
require 'csvigo'
require 'nn'
require 'optim'
require 'xlua'

features = {"M12", "M11", "M10", "M9", "M8", "M7", "M6", "M5", "M4", "M3", "M2",
            "M1", "D15", "D14", "D13", "D12", "D11", "D10", "D9", "D8", "D7",
            "D6", "D5", "D4", "D3", "D2", "D1", "Jan"}
label = "Label"
info = {"Ticker", "Month", "Year"}

opt = {
  learningRate = 1e-3,
  learningRateDecay = 1e-7,
  batchSize = 1,
  weightDecay = 0,
  momentum = 0,
  maxIter = 2,
}

-- Load training data.
loaded = csvigo.load{path="data/trainData.dat"}

featTable = {}
for key, value in pairs(features) do
  for ind, v in pairs(loaded[value]) do
    loaded[value][ind] = tonumber(v)
  end
  featTable[key] = loaded[value]
end

labTable = loaded[label]
for ind, v in pairs(labTable) do
  labTable[ind] = tonumber(v)
end

featTensor = torch.Tensor(featTable):transpose(1,2)
labTensor = torch.Tensor(labTable)

trainData = {
   data = featTensor:float(),
   labels = labTensor,
   size = 24534
}

-- Load test data.
loaded = csvigo.load{path="data/testData.dat"}

featTable = {}
for key, value in pairs(features) do
  for ind, v in pairs(loaded[value]) do
    loaded[value][ind] = tonumber(v)
  end
  featTable[key] = loaded[value]
end

labTable = loaded[label]
for ind, v in pairs(labTable) do
  labTable[ind] = tonumber(v)
end

featTensor = torch.Tensor(featTable):transpose(1,2)
labTensor = torch.Tensor(labTable)

testData = {
   data = featTensor:float(),
   labels = labTensor,
   size = 8250
}

-- Preprocess each input dimension by standardization.
mean = {}
std = {}
for i = 1, 27 do
  mean[i] = trainData.data[{ {}, i }]:mean()
  std[i] = trainData.data[{ {}, i }]:std()
  trainData.data[{ {},i }]:add(-mean[i])
  trainData.data[{ {},i }]:div(std[i])
end

-- Standardize test data using training data mean and std.
for i = 1, 27 do
  testData.data[{ {},i }]:add(-mean[i])
  testData.data[{ {},i }]:div(std[i])
end

-- Define model.
ninputs = 28
noutputs = 2
nnsize = {ninputs, 500, 500, 1000, noutputs}
model = nn.Sequential()
-- model:add(nn.Dropout(0.5))
model:add(nn.Linear(nnsize[1],nnsize[2]))
model:add(nn.ReLU())
-- model:add(nn.Dropout(0.5))
model:add(nn.Linear(nnsize[2],nnsize[3]))
model:add(nn.ReLU())
-- model:add(nn.Dropout(0.5))
model:add(nn.Linear(nnsize[3],nnsize[4]))
model:add(nn.ReLU())
model:add(nn.Linear(nnsize[4],nnsize[5]))

-- Define loss model.
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

-- Define training tools, parameters and methods.
classes = {'0','1'}

confusion = optim.ConfusionMatrix(classes)

trainLogger = optim.Logger(paths.concat('results', 'train.log'))

testLogger = optim.Logger(paths.concat('results', 'test.log'))

parameters,gradParameters = model:getParameters()

optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay
}
optimMethod = optim.sgd

-- Define training procedure.
function train()

  -- track epoch
  epoch = epoch or 1

  -- local time to track training time
  local time = sys.clock()

  -- set model to training mode
  model:training()

  -- shuffle training data index at each epoch
  shuffle = torch.randperm(trainData.size)

  -- do one epoch
  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  for t = 1, trainData.size, opt.batchSize do

    -- disp progress
    xlua.progress(t, trainData.size)

    -- create mini batch
    local inputs = {}
    local targets = {}
    for i = t, math.min(t+opt.batchSize-1, trainData.size) do
      -- load new sample
      local input = trainData.data[shuffle[i]]
      local target = trainData.labels[shuffle[i]]
      input = input:double()
      table.insert(inputs, input)
      table.insert(targets, target)
    end

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end

      -- reset gradients
      gradParameters:zero()

      -- f is the average of all criterions
      local f = 0

      -- evaluate function for complete mini batch
      for i = 1,#inputs do
        -- estimate f
        local output = model:forward(inputs[i])
        local err = criterion:forward(output, targets[i])
        f = f + err
        -- estimate df/dW
        local df_do = criterion:backward(output, targets[i])
        model:backward(inputs[i], df_do)
        -- update confusion
        confusion:add(output, targets[i])
      end

      -- normalize gradients and f(X)
      gradParameters:div(#inputs)
      f = f/#inputs
      -- return f and df/dX
      return f,gradParameters

    end

    -- optimize on current mini-batch
    if optimMethod == optim.asgd then
      _,_,average = optimMethod(feval, parameters, optimState)
    else
      optimMethod(feval, parameters, optimState)
    end
  end

  -- time taken
  time = sys.clock() - time
  time = time / trainData.size
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)

  -- update logger/plot
  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  if opt.plot then
    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    trainLogger:plot()
  end

  -- save/log current net
  local filename = paths.concat('results', 'model.net')
  os.execute('mkdir -p ' .. sys.dirname(filename))
  print('==> saving model to '..filename)
  torch.save(filename, model)

  -- next epoch
  confusion:zero()
  epoch = epoch + 1

end

-- Define testing procedure.
-- test function
function test()

  -- local vars
  local time = sys.clock()

  -- averaged param use?
  if average then
    cachedparams = parameters:clone()
    parameters:copy(average)
  end

  -- set model to evaluate mode
  model:evaluate()

  -- test over test data
  print('==> testing on test set:')
  for t = 1,testData.size do

    -- disp progress
    xlua.progress(t, testData.size)

    -- get new sample
    local input = testData.data[t]
    local target = testData.labels[t]
    input = input:double()

    -- test sample
    local pred = model:forward(input)
    confusion:add(pred, target)

  end

  -- timing
  time = sys.clock() - time
  time = time / testData.size
  print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)

  -- update log/plot
  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}

  -- averaged param use?
  if average then
    -- restore parameters
    parameters:copy(cachedparams)
  end

  -- next iteration:
  confusion:zero()

end

print '==> training!'
while true do
  train()
  test()
end
