path1 <-  rstudioapi::getActiveDocumentContext()$path 
path2 <- gsub("codes","data",path1)
setwd(dirname(path2))

library(kknn)

data <-read.csv("ResNet18_PlantDisease_45K_Values.csv")
label <- read.csv("ResNet18_PlantDisease_45K_Labels.csv")
s1 <- specClust(data,centers = 200, nn = 30)

#Append the result of clustering
label$Spec200 <- s1$cluster


write.csv(label,"ResNet18_PlantDisease_45K_tSNE_Spec.csv",row.names = FALSE)




