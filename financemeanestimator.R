library(mxnet)
library(mlbench)
data(BostonHousing, package="mlbench")

train.ind = seq(1, 506, 1)
train.x = data.matrix(BostonHousing[train.ind, -14])
train.y = BostonHousing[train.ind, 14]
test.x = data.matrix(BostonHousing[-train.ind, -14])
test.y = BostonHousing[-train.ind, 14]

data <- mx.symbol.Variable("data")
numtot <- 10
numep <- 100
train.ind = seq(1,numtot*numep,1)
train.x = numeric(200)
train.y = numeric(200)
test.x = 
test.y = 
es = 1
es.t = numeric(200) 
mu.t = numeric(200) 

for (i in 1:numtot){
  mu <- runif(1, min = -1, max = 1)
  
  for (z in 1:numep){
    x <- rnorm(200, mu, 1) 
    train.x = append(train.x,x)
    train.y = append(train.y,mu)
    es <- 1/200 * sum(x)
    es.t <- append(es.t,es)
    mu.t <- append(mu.t,mu)
  }
}
error <- sqrt((es.t-mu.t)*(es.t-mu.t)) 


fc1 <- mx.symbol.FullyConnected(data, num_hidden=50)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type='sigmoid')
fc2 <- mx.symbol.FullyConnected(act1, num_hidden=1)
# Use linear regression for the output layer
lro <- mx.symbol.LinearRegressionOutput(fc2)

mx.set.seed(0)
model <- mx.model.FeedForward.create(lro, X=train.x, y=train.y,
                                     ctx=mx.gpu(), num.round=1, array.batch.size=200,
                                     learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse)

preds = predict(model, test.x)
sqrt(mean((preds-test.y)^2))
demo.metric.mae <- mx.metric.custom("mae", function(label, pred) {
  res <- mean(abs(label-pred))
  return(res)
})
mx.set.seed(0)
model <- mx.model.FeedForward.create(lro, X=train.x, y=train.y,
                                     ctx=mx.cpu(), num.round=50, array.batch.size=20,
                                     learning.rate=2e-6, momentum=0.9, eval.metric=demo.metric.mae)