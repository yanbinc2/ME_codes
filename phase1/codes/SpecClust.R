path1 <-  rstudioapi::getActiveDocumentContext()$path 
path2 <- gsub("codes","data",path1)
setwd(dirname(path2))

library(kknn)

f1 <-read.csv("ResNet18_PlantDisease_45K_Values.csv")
f2 <- read.csv("ResNet18_PlantDisease_45K_Labels.csv")
s1 <- specClust(f1,200)
f2$Spec200 <- s1$cluster


write.csv(f2,"ResNet18_PlantDisease_45K_tSNE_Spec.csv",row.names = FALSE)


h1 <-read.csv("VGG16_PlantDisease_45K_Values.csv")
h2 <- read.csv("VGG16_PlantDisease_45K_Labels.csv")
s2 <- specClust(h1,200)
h2$Spec200 <- s2$cluster


write.csv(h2,"VGG16_PlantDisease_45K_tSNE_Spec.csv",row.names = FALSE)



