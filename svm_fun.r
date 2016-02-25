# Writing a simple SVM algorithm from scratch
# we should not get 76 percents otherwise overfit
rm(list=ls()) # clear all variables
setwd('~/Dropbox/applied_ml/hw2')
wdat<-read.csv('adult.data', header=FALSE, na.strings='?')
library(caret)

prepData<-wdat[, -c(2,4,6,7,8,9,10,14)]
completeData<-na.omit(prepData)
bigx<-scale(completeData[,-7]) #feature scaling 

bigy<-as.matrix(completeData[7])

bigy<-sapply(bigy, function (x) switch(x, ' >50K'=1, ' <=50K'=-1))

partition<-createDataPartition(y=bigy, p=.8, list=FALSE)
trainx<-bigx[partition,]
# pretrainx<-bigx[partition,]
# trainx<-scale(pretrainx)
trainy<-bigy[partition]

# prerestx<-bigx[-partition,]
# restx<-scale(prerestx)
restx<-bigx[-partition,]
resty<-bigy[-partition]

partition2<-createDataPartition(y=resty, p=.5, list=FALSE)
testx<-restx[partition2,]
testy<-resty[partition2]

validx<-restx[-partition2,]
validy<-resty[-partition2]
#a<-c(1,1,1,1,1,1)
a<-c(0,0,0,0,0,0)
b<-0
#b<-1

plot(x=NULL, y=NULL, xlim=c(1,500), ylim=c(0,1), type='n', xlab='#30steps', ylab='accuracy')

colors<-rainbow(5)
lambdalist<-c(1, 0.1, 0.01, 0.001, 0.00001)


acc_list = c(0,0,0,0,0)
for (i in 1:5){
  #store the sum of accuracy to compute average
  ysum<-0
  #x value in the plot--#30 steps
  round<-1
  #if the first point of a plot
  first<-TRUE
  for (e in 1:50){
    held_id<-createDataPartition(y=trainy, p=.002, list=FALSE)
    trainx_e<-trainx[-held_id,]
    trainy_e<-trainy[-held_id]
    held_x<-trainx[held_id,]
    held_y<-trainy[held_id]
    # turn validation data into a matrix so that no need to do that in the for loop
    xheld_matrix<-as.matrix(held_x)
    #check!
    held_len<-nrow(held_x)
    
    s_size<-1/(0.01*e+50)  		
    # sizevector<-c(sizevector, s_size)
    
    for (st in 1:300){
      k<-sample(1:nrow(trainx_e), 1, replace=TRUE)
      y_hat<-sum(a*t(trainx_e[k,]))+b
      if (trainy_e[k]*y_hat>=1){
        a<-a - s_size*(lambdalist[i])*a
      }
      else{
        a<- a - s_size*(lambdalist[i]*a-trainy_e[k]*trainx_e[k,])
        b<- b + s_size*trainy_e[k]
      }
      
      # avector<-c(avector, a)
      # bvector<-c(bvector, b)
      
      if (st%%30==0){
        #predicted y
        y_pred_vector<- xheld_matrix%*%as.numeric(a) + b*rep(1, held_len)
        res<-held_y*y_pred_vector
        held_acc<-sum(res>0)/length(res)
        ysum<-ysum+held_acc
        if (first){
          first<-FALSE
        }
        else{
          #plot individually
          segments(round-1, prey, round, held_acc, col=colors[i])
          #plot average
          #segments(round-1, prey, round, ysum/round, col=colors[i])
        }
        #plot individually
        prey<-held_acc
        #plot average
        #prey<-ysum/round
        round<-round+1
      }
    }
   
    
  }
  #compute the acc of different lambda runs on the validation set
  y_pred_val<-validx%*%as.numeric(a)+b*rep(1, nrow(validx))
  res_val<-validy*y_pred_val
  acc_for_lambda<-sum(res_val>0)/length(res_val)
  acc_list[i]<-acc_for_lambda
}



##################################################################
# Now from our accuracy on the validation set we found that 
# lambda=0.01 has the best accuracy on the validation set consistently
# So we will use this parameter to get the test-accuracy

#setting up an experiment loop. We will average over 10 different 
#train-test splits
test_acc<-array(dim=10)
for(j in 1:10)
{
  #splitting the train, test split to 90 percents train 10 percents test
  train_id<-createDataPartition(y=bigy, p=0.9, list=FALSE)
  x_train<-bigx[train_id, ]
  y_train<-bigy[train_id]
  x_test<-bigx[-train_id, ]
  y_test<-bigy[-train_id]   	
  weight=c(0,0,0,0,0,0)
  intercept = 0
  lambda=0.01
  for (e in 1:50){
    s_size<-1/(0.01*e+50) 
    for (st in 1:300){
      k<-sample(1:nrow(x_train), 1, replace=TRUE)
      y_hat<-sum(weight*t(x_train[k,]))+intercept
      if (y_train[k]*y_hat>=1){
        weight<-weight - s_size*(lambda)*weight
      }
      else{
        weight<- weight - s_size*(lambda*weight-y_train[k]*x_train[k,])
        intercept<- intercept + s_size*y_train[k]
      }
    }

  }
  y_pred_test<-x_test%*%as.numeric(weight)+intercept*rep(1, nrow(x_test))
  result<-y_test*y_pred_test
  acc<-sum(result>0)/length(result)
  test_acc[j]<-acc
}
ave_test_acc<-sum(test_acc)/length(test_acc)
print(ave_test_acc)