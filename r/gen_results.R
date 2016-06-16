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

results <- fread("NEWEST_RES/results.tsv",header=F,sep="\t")
setnames(results,c("iter","model","perpl","rank"))
results[rank == 0,mean(perpl),by="model"]
assoc <- results[rank == 0,mean(perpl),by="model"]
uv <- assoc[model=='user_assoc']$V1
ov <- assoc[model=="our_assoc"]$V1
(ov-uv)/uv

sent <- results[rank != 0,mean(rank),by="model"]
uv <- sent[model=='u_sent']$V1
ov <- sent[model=="our_sent"]$V1
(ov-uv)/uv


output_dir <- "~/Dropbox/Kenny/papers/current/defense/study_2/acl_sub/"

eta <- data.table(npyLoad("NEWEST_RES/900_assoc_eta.npy"))
phi <- data.table(npyLoad("NEWEST_RES/900_sent_phi.npy"))
assoc_mu <- data.table(npyLoad("NEWEST_RES/900_assoc_mu.npy"))
sent_mu <- data.table(npyLoad("NEWEST_RES/900_sent_mu.npy"))
sent_mu_0 <- data.table(npyLoad("NEWEST_RES/sent_mu_0.npy"))
sent_labels <- fread("NEWEST_RES/sent_ids_list.txt",header=F)
sent_labels$sent_index <- 1:nrow(sent_labels)
assoc_sig <- data.table(cov2cor(npyLoad("NEWEST_RES/900_assoc_sigma.npy")))
index_to_identity <- fread("NEWEST_RES/index_to_identity_final.tsv",header=F)
index_to_id <- fread("NEWEST_RES/index_to_id_final.tsv",header=F)


id_data <- merge(index_to_id,index_to_identity,by="V1")
setnames(id_data, c("index","id","name"))

sentdf <- data.table(mu=sent_mu$V1, mu0 = sent_mu_0$V1,label=sent_labels$V1,sent_ind=sent_labels$sent_index)
sentdf[,diff_abs:=abs(mu-mu0)]
sentdf[,diff:=mu-mu0]
sentdf$id <- sapply(sentdf$label, function(x){substr(x, 0, nchar(x)-1)})
sentdf$epa <-  sapply(sentdf$label, function(x){substr(x, nchar(x), nchar(x))})
sentdf <- merge(sentdf, id_data,by="id")
#sentdf$label <- NULL

sentdf <- arrange(sentdf,sent_ind)
sentdf$sent_name <- paste(sentdf$name, sentdf$epa,sep="_")
setnames(phi,sentdf$sent_name)

epa_dat <- spread(sentdf[,c("name","epa","mu"),with=F], epa, mu)
epa_dat$a_scale <- scale(epa_dat$a)
epa_dat$e_scale <- scale(epa_dat$e)
epa_dat$p_scale <- scale(epa_dat$p)


identities <- arrange(id_data,index)$name
setnames(eta, index_to_identity$V2)

##### FULL PAIRWISE DATA
colnames(assoc_sig) <- colnames(eta)
assoc_sig$name  <- colnames(eta)  
m <- melt(assoc_sig,id="name")
m <- m[as.character(m$name) < as.character(m$variable)]
m <- merge(m,epa_dat[,c("name","e","p","a"),with=F],by.x="name",by.y="name")
setnames(m,c("e","p","a"),c("eN","pN","aN"))
m <- merge(m,epa_dat[,c("name","e","p","a"),with=F],by.x="variable",by.y="name")
setnames(m,c("e","p","a"),c("eV","pV","aV"))
m[,sent_diff := sqrt( (eN-eV)^2 + (pN-pV)^2+ (aN-aV)^2)] #
m[,sent_sim := 1/(1+sent_diff)]

#m_no_bad <- m[!name %in% bad_words & !variable %in% bad_words,]

#############################

epa_dat$prob <- with(epa_dat,
                      interp.surface(kde2d(e,p), 
                                     data.frame(x=e,y=p)))
p1 <- ggplot(epa_dat, aes(e,p,color=a,
                           label=ifelse(prob < .025 | 
                                          (abs(a) > 1 & prob < .07),
                                        name,NA)))  
p1 <- p1 + geom_hline(yintercept=0) + geom_vline(xintercept=0) 
p1 <- p1 + scale_color_gradient2("Activity", low='blue',
                                 mid='light grey',
                                 high=muted('red'))  
p1 <- p1 + geom_point(size=3) + geom_text_repel(color='black')
p1 <- p1 + xlab("Evaluative (Good/Bad)") + ylab("Potency (Strong/Weak)")
ggsave(file.path(output_dir,"affective_plot.jpg"),dpi=600,h=6,w=8)

mean(sentdf[name %in% c("police", "cop", "police officer") & epa == 'e']$diff)
mean(sentdf[name %in% c("police", "cop", "police officer") & epa == 'p']$diff)
mean(sentdf[name %in% c("police", "cop", "police officer") & epa == 'a']$diff)

sentdf[name =='protester' & epa == 'e']$diff
sentdf[name =='protester' & epa == 'p']$diff
sentdf[name =='protester' & epa == 'a']$diff


q <- sentdf[name %in% c("police","cop","police officer","protester") & epa != "a"]
q$epa <- factor(q$epa, levels=c("e","p"),
                labels=c("Evaluative","Potency"))
ggplot(q, aes(name,mu-mu0,fill=mu-mu0 > 0)) + geom_bar(stat='identity') + facet_wrap(~epa,scales="free_y") + theme(legend.position="none") + ylab("Change in value between\nmodel estimate and survey data") + xlab("") + scale_fill_manual(values=c("red","blue"))
############ Semantic Stuff ###################

run_huge <- function(data, names_dat,l=.3){
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
  jpeg(paste0(output_dir,"infomap_clustering_",l,".jpeg"), units="in", width=12,height=7,res=600,quality=600)
  plot(lv,gf,vertex.size=4,vertex.label=NA,mark.groups=NULL,edge.color='grey')
  dev.off()
  write.csv(arrange(data.frame(name=lv$names,membership=lv$membership),membership),paste0("group_",l,".csv"))
  return(lv)
}

######### CLUSTERING ############
lv3 <- run_huge(eta, names(eta), .3)
lv45 <- run_huge(eta, names(eta),.45)
lv6 <- run_huge(eta, names(eta),.6)

grouping <- arrange(data.frame(name=lv3$names,membership=lv3$membership),membership)
write.table(grouping, "infomap_clusters.tsv",row.names=F,quote=F,sep="\t")

### READ IT BACK IN AFTER LABELING
grouping_with_names <- fread("infomap_clusters_with_names.tsv",sep="\t")
grouping_with_names$membership <- NULL

h_w_group <- merge(h_no_bad,grouping_with_names, by.x="name",by.y="name")
setnames(h_w_group, "group_name","from_group")
h_w_group <- merge(h_w_group,grouping_with_names, by.x="variable",by.y="name")
setnames(h_w_group, "group_name","to_group")
#################################


####### THUG EXPLORATION ############
#prob == .8 for thug

identity <- "thug"
thug_dat <- m[name ==identity | variable==identity]
thug_dat$thug_name <- with(thug_dat, ifelse(name ==identity,as.character(variable),name ))
z <- kde2d(thug_dat$sent_sim,thug_dat$value)
thug_dat$prob <- interp.surface(z, data.frame(x=thug_dat$sent_sim,y=thug_dat$value))
thug_dat[thug_name=='nigga']$thug_name <- 'n$$$a'

p <- ggplot(thug_dat, aes(value,sent_sim,label=ifelse(prob < .8,thug_name,NA)))
p <- p + geom_hline(yintercept=mean(thug_dat$sent_sim),color='grey') 
p <- p + geom_vline(xintercept=0,color='grey') + geom_point(alpha=.8,color=muted('blue'))
p <- p + geom_label_repel(size=6)
p <- p + xlab("Semantic Relationship Strength") +ylab("Affective Similarity")
theme_set(theme_bw(20))
ggsave(file.path(output_dir,"thug.jpeg"),p,dpi=600,h=6,w=8)




