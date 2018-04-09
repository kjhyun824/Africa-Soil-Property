library(nnet)
library(caret)
library(plyr)
setwd("C:/Users/samsung/Desktop/")
data <-read.csv("Rdata for NN.csv") #all data
#data <-read.csv("Normalization_NN.csv")
x<- data[1:18]               #input
y <- data[19:24]            #output
#y[1] <-as.character(y[1])
data <-data.frame(y[1],x) #1:gen3, 2:gen5, 3:gen10, 4:TC3, 5:TC5, 6:TC10 (input + output) we are using

k=10
data$id <-sample(1:k,nrow(data),replace=TRUE)
list <-1:k
prediction<-data.frame()
testsetCopy <- data.frame()

#Creating a progress bar to know the status of CV


for (i in 1:k)
{  
    trainingset <-subset(data, id%in%list[-i])
    testset <-subset(data,id%in%c(i))
    trainingset <- trainingset[,-2]
    testset<-testset[,-2] 
 
    trainingset$gen3 <- as.factor(trainingset$gen3)
    testset$gen3 <-as.factor(testset$gen3)
    
    trainingset.scale = cbind(trainingset[1],scale(trainingset[-1]))
    apply(trainingset.scale[-1],2,sd)
    testset.scale = cbind(testset[1],scale(testset[-1]))
    apply(testset.scale[-1],2,sd)
    
    ANN<-nnet(trainingset.scale$gen3~.,data=trainingset.scale,size=10,maxit=200) 
    predicted = predict(ANN,testset.scale[,-1],type="class")
   # temp <- as.data.frame(predicted)
   
   # prediction <-rbind(prediction,predicted) 
   #  testsetCopy <-rbind(testsetCopy, testset.scale[,1])
    prediction <- as.matrix(predicted)
    testsetCopy <-as.matrix(testset.scale[,1])
    model.confusion.matrix = table(testsetCopy, prediction)
    print(model.confusion.matrix)
    
}  

#prediction <-as.matrix(prediction)
#testsetCopy <-as.matrix(testsetCopy)

model.confusion.matrix = table(testsetCopy, prediction)
