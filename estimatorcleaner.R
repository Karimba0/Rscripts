library(mxnet)


data <- mx.symbol.Variable("data")
numtot <- 100
numep <- 100
train.ind = seq(1,numtot*numep,1)
train.x = array(rep(0,numtot*numep*200),c((numtot*numep),200))
train.y = numeric(1)
test.x = rnorm(200, 3, 1)
test.y = rep(3,200)
  es = 1
es.t = numeric(1) 
mu.t = numeric(numtot*numep) 

for (i in 1:numtot){
  mu <- runif(1, min = 0, max = 1)
  
  for (z in 1:numep){
    x <- rnorm(200, mu, 1) 
    train.x[i*z,] = x
    train.y[i*z] = mu
    es <- 1/200 * sum(x)
    es.t <- append(es.t,es)
    mu.t <- append(mu.t,mu)
  }
}
error <- sqrt((es.t-mu.t)^2) 
print(mean(error))
train.x = data.matrix(train.x)
data <- mx.symbol.Variable("data")


fc1 <- mx.symbol.FullyConnected(data, num_hidden=256)
act1 = mx.symbol.Activation(fc1, name='tanh1', act_type='tanh')
fc2 <- mx.symbol.FullyConnected(act1, num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='tanh2', act_type='tanh')
fc2 <- mx.symbol.FullyConnected(act1, num_hidden=1)
# Use linear regression for the output layer
lro <- mx.symbol.LinearRegressionOutput(fc2)

mx.set.seed(0)
model <- mx.model.FeedForward.create(lro, X=train.x, y=train.y,
                                     ctx=mx.gpu(), num.round=100, array.batch.size=20,
                                     learning.rate=2e-2, momentum=0.9, eval.metric=mx.metric.rmse)

preds = predict(model, test.x)
print(sqrt(mean((preds-test.y)^2)))
