library(data.table)
library(ggplot2)
theme_set(theme_bw(20))

d <- lapply(list.files("random_split_output/",full.names=T), fread)

dat <- data.frame()
for(i in 1:length(d)){
  x <- d[[i]]
  x$run <- i
  dat <- rbind(dat,x)
}

v <- function(x){
  v1 <- x[V2 == "our_assoc"]$V3
  v2 <- x[V2 == "user_assoc"]$V3
  return( abs(v1-v2)/v2)
}
  
k <- function(x){
  v1 <- x[V2 == "our_sent"]$V4
  v2 <- x[V2 == "u_sent"]$V4
  return( abs(v1-v2)/v2)
}

          
dat[,v(.SD),by="run"]
mean(dat[,v(.SD),by="run"]$V1)
dat[,k(.SD),by="run"]
mean(dat[,k(.SD),by="run"]$V1)
