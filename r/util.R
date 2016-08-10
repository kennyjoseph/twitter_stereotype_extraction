library(devtools)  
source_url("https://raw.githubusercontent.com/ggrothendieck/gsubfn/master/R/list.R")
library(huge)
library(dplyr)
library(tidyr)
library(reshape2)
library(data.table)
library(ggplot2)
library(RcppCNPy)
library(tidyr)
library(ggiraph)
library(Hmisc)
library(ggrepel)
library(stringr)
library(fields)
library(scales)


output_dir <- "~/Dropbox/Kenny/papers/submitted/cscw_sub/figs2/"

get_m <- function(dirname, n){
  
  eta <- data.table(npyLoad(file.path(dirname,paste0(n,"_assoc_eta.npy"))))
  assoc_sig <- data.table(cov2cor(npyLoad(file.path(dirname,paste0(n,"_assoc_sigma.npy")))))
  index_to_identity <- fread(file.path(dirname,"index_to_identity_final.tsv"),header=F)
  index_to_id <- fread(file.path(dirname,"index_to_id_final.tsv"),header=F)
  id_data <- merge(index_to_id,index_to_identity,by="V1")
  setnames(id_data, c("index","id","name"))
  identities <- arrange(id_data,index)$name
  setnames(eta, index_to_identity$V2)
  colnames(assoc_sig) <- colnames(eta)
  assoc_sig$name  <- colnames(eta)  
  m <- melt(assoc_sig,id="name")
  #m <- m[as.character(m$name) < as.character(m$variable)]
  return(list(m,eta))
}

run_huge <- function(data, names_dat,l=.3,suffix=""){
  opt_h <- huge(as.matrix(data),method="glasso",lambda =l,cov.output=T)
  df <- data.frame(as.matrix(cov2cor(opt_h$cov[[1]])))
  names(df) <- names_dat
  df$name <-  names_dat
  df <- melt(df, id.vars=c("name"))
  df[df$variable == df$name,]$value <- NA
  df <- data.table(df)
  df[abs(value) < .01,]$value <- NA
  df <- arrange(df[as.character(name) < as.character(variable)],-value)
  df <- df[!is.na(value)]
  df <- data.table(df)
  gf <- graph_from_data_frame(df[value > 0,c("name","variable"),with=F], directed = FALSE)
  write.csv(df[value > 0,c("name","variable","value"),with=F], paste0("net_",l,".csv"))
  
  E(gf)$weight <- df[value > 0,]$value
  lv <- cluster_louvain(gf)
  jpeg(paste0(output_dir,"infomap_clustering_",l,suffix,".jpeg"), units="in", width=12,height=7,res=600,quality=600)
  plot(lv,gf,vertex.size=4,vertex.label=NA,mark.groups=NULL,edge.color='grey')
  dev.off()
  
  jpeg(paste0(output_dir,"infomap_clustering_labels_",l,suffix,".jpeg"), units="in", width=18,height=12,res=600,quality=600)
  plot(lv,gf,vertex.size=3,mark.groups=NULL,edge.color='grey',vertex.frame.color=NA)
  dev.off()
  
  write.csv(arrange(data.frame(name=lv$names,membership=lv$membership),membership),paste0("group_",l,".csv"))
  return(list(lv,df,gf))
}
