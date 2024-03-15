path1 <-  rstudioapi::getActiveDocumentContext()$path 
path2 <- gsub("codes","data",path1)
setwd(dirname(path2))

library(kknn)

f1 <-read.csv("MNIST_tSNE_5000.csv")
f2 <- read.csv("MNIST_Labels_5000.csv")
s1 <- specClust(f1,20)
f2$Spec20 <- s1$cluster


write.csv(f2,"MNIST_Labels_Spec20.csv",row.names = FALSE)
