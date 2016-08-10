
source("util.R")
############# GENERATE RESULTS  OF EVAL #############

results <- fread("NEWEST_RES/results_only_rank.tsv",header=F,sep="\t")
setnames(results,c("iter","model","perpl","rank"))
sent <- results[,mean(rank),by="model"]
uv <- sent[model=='u_sent']$V1
ov <- sent[model=="our_sent"]$V1
(ov-uv)/uv

################ GET DATA ###########################

eta <- data.table(npyLoad("NEWEST_RES/900_assoc_eta.npy"))
phi <- data.table(npyLoad("NEWEST_RES/900_sent_phi.npy"))
assoc_mu <- data.table(npyLoad("NEWEST_RES/900_assoc_mu.npy"))
sent_mu <- data.table(npyLoad("NEWEST_RES/900_sent_mu.npy"))
sent_mu_0 <- data.table(npyLoad("NEWEST_RES/sent_mu_0.npy"))
sent_sigma <- data.table(npyLoad("NEWEST_RES/900_sent_sigma.npy"))

sent_labels <- fread("NEWEST_RES/sent_ids_list.txt",header=F)
sent_labels$sent_index <- 1:nrow(sent_labels)
assoc_sig <- data.table(cov2cor(npyLoad("NEWEST_RES/900_assoc_sigma.npy")))
index_to_identity <- fread("NEWEST_RES/index_to_identity_final.tsv",header=F)
index_to_id <- fread("NEWEST_RES/index_to_id_final.tsv",header=F)


id_data <- merge(index_to_id,index_to_identity,by="V1")
setnames(id_data, c("index","id","name"))

sentdf <- data.table(mu=sent_mu$V1, mu0 = sent_mu_0$V1,sigma=diag(as.matrix(sent_sigma)),
                     label=sent_labels$V1,sent_ind=sent_labels$sent_index)
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

epa_sigma_dat <- spread(sentdf[,c("name","epa","sigma"),with=F], epa, sigma)

identities <- arrange(id_data,index)$name
setnames(eta, index_to_identity$V2)

############################## SEMANTIC STUFF #############################

list[m,eta] <- get_m("NEWEST_RES",900)
list[m_rand,eta_rand] <- get_m("RAND_RES/",300)

######### CLUSTERING ############
list[lv3,df3,gf3] <- run_huge(eta, names(eta), .3)
list[lv45,df45,gf45] <- run_huge(eta, names(eta),.45)
list[lv6,df6,gf6] <- run_huge(eta, names(eta),.6)
print(xtable(data.table(name=lv45$names,membership=lv45$membership)[,paste(name,collapse=", "),by="membership"]),include.rownames=F)

list[lvr4,dfr4,gfr4] <- run_huge(eta_rand, names(eta_rand),.38,"RAND")
print(xtable(data.table(name=lvr5$names,membership=lvr5$membership)[,paste(name,collapse=", "),by="membership"]),include.rownames=F)

#################################

V(gf45)$group <- lv45$membership
k <- components(gf45)
ferg <-  delete_vertices(gf45,names(k$membership[k$membership != 1]))
V(ferg)$bet <- betweenness(ferg)
p <- ggplot(ferg, aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_edges( color = "grey50",size=.2,alpha=.3) +geom_nodes(aes(color=factor(group),size=bet),alpha=1) +
  theme_blank() 
p <- p + geom_nodetext_repel(data = function(x) { x[ x$vertex.names %in% of_interest, ]},aes(label=vertex.names),size=3.5) 
#p <- p + theme(legend.position='none')
ggsave(file.path(output_dir,"fergnet.jpg"),p, dpi=600,h=7,w=8)

rand <-  delete_vertices(gfr4,! V(gfr4)$name %in% V(ferg)$name)
g <- data.table(name=lv45$names,membership=lv45$membership)
setkey(g,"name")
V(rand)$group <- g[V(rand)$name]$membership
p1 <- p %+% ggnetwork(rand)
p1
ggsave(file.path(output_dir,"randnet.jpg"),p1, dpi=600,h=8,w=8)

arrange(dfr4[name=='racist' | variable =='racist'],-value)[1:6]
arrange(df45[name=='racist' | variable =='racist'],-corr)


############################# SENTIMENT STUFF #############################

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
ggsave(file.path(output_dir,"affective_plot.jpg"),p1,dpi=600,h=6,w=8)

of_interest <- grouping_with_names[V3 %in% c("legal","race")]$name
q <- sentdf[name %in% of_interest & epa !="a" ]
q$epa <- factor(q$epa, levels=c("e","p"),
                labels=c("Evaluative","Potency"))
q <- merge(q,grouping_with_names,by="name")
p2 <- ggplot(q, aes(name,mu,ymin=mu-sigma,ymax=mu+sigma,color=V3)) 
p2 <- p2 + geom_pointrange() + facet_wrap(~epa)  + geom_point(aes(x=name,y=mu0),size=1.5,color='red') 
p2 <- p2 + coord_flip() + xlab("") + ylab("Affective Value [-4.3,4.3]")
p2 <- p2 + scale_color_manual("Identity\nCluster",values=c("black","blue"))
ggsave(file.path(output_dir,"povprot.jpg"),p2, dpi=600,h=9,w=10)


####################### FULL PAIRWISE DATA ##########################
colnames(assoc_sig) <- colnames(eta)
assoc_sig$name  <- colnames(eta)  
m <- melt(assoc_sig,id="name")
m <- m[as.character(m$name) < as.character(m$variable)]
m <- merge(m,epa_dat[,c("name","e","p","a"),with=F],by.x="name",by.y="name")
setnames(m,c("e","p","a"),c("eN","pN","aN"))

m <- merge(m,epa_sigma_dat[,c("name","e","p","a"),with=F],by.x="name",by.y="name")
setnames(m,c("e","p","a"),c("eSigN","pSigN","aSigN"))

m <- merge(m,epa_dat[,c("name","e","p","a"),with=F],by.x="variable",by.y="name")
setnames(m,c("e","p","a"),c("eV","pV","aV"))

m <- merge(m,epa_sigma_dat[,c("name","e","p","a"),with=F],by.x="variable",by.y="name")
setnames(m,c("e","p","a"),c("eSigV","pSigV","aSigV"))

m[,sent_sim := 1/(1+ sqrt( (eN-eV)^2 + (pN-pV)^2+ (aN-aV)^2))]
m$bdistE <- 1/(1+sqrt(m[,bhattacharyya.dist(eN,eV,eSigN,eSigV),by=c("name","variable")]$V1))
m$bdistP <- 1/(1+sqrt(m[,bhattacharyya.dist(pN,pV,pSigN,pSigV),by=c("name","variable")]$V1))
m$bdistA <- 1/(1+sqrt(m[,bhattacharyya.dist(aN,aV,aSigN,aSigV),by=c("name","variable")]$V1))

####### THUG EXPLORATION ############
#prob == .8 for thug

identity <- "thug"
thug_dat <- m[name ==identity | variable==identity]
thug_dat$thug_name <- with(thug_dat, ifelse(name ==identity,as.character(variable),name ))
setnames(thug_dat,"value","corr")
thug_dat <- melt(thug_dat,c("thug_name","corr"),c("sent_sim","bdistE","bdistP","bdistA"))
thug_dat[thug_name=='nigga']$thug_name <- 'n$$$a'

get_prob <- function(dat){
  z <- kde2d(dat$corr,dat$value)
  interp.surface(z, data.frame(x=dat$corr,y=dat$value))
}
thug_dat[,prob:=get_prob(.SD),by=c("variable")]

thug_dat$variable <- factor(thug_dat$variable, levels=c("sent_sim","bdistE","bdistP","bdistA"),
                            labels=c("Overall","Evaluative Only","Potency Only","Activity Only"))

p <- ggplot(thug_dat, aes(corr,value,label=ifelse(prob < .8,thug_name,NA)))+ facet_wrap(~variable,scales="free_y") 
p <- p  + geom_point(alpha=.5,color=muted('blue'))
p <- p + geom_hline(data=thug_dat[,mean(value),by=variable],aes(yintercept=V1),color='grey')
p <- p + geom_vline(xintercept=0,color='grey')
p <- p + geom_text_repel(size=4)
p <- p + xlab("Semantic Relationship Strength") +ylab("Affective Similarity")
theme_set(theme_bw(20))
ggsave(file.path(output_dir,"thug.jpeg"),p,dpi=600,h=6,w=8)






of_interest <- c("officer", "attorney", "police", "deputy", "judge", "lawyer", "chief", "juror", "gunman", "protester", "inmate", "prosecutor", "survivor", "cop", "firefighter", "criminal", "victim", "police officer", "sheriff", "hostage", "protestor", "witness", "prisoner", "shooter")
of_interest <- c(of_interest,c("black", "black woman", "white man", "black man", "white", "african american", "racist", "white woman"))
ferg2 <-  delete_vertices(gf3, !V(gf3)$name %in% of_interest)
V(ferg2)$e_val <- sapply(V(ferg2)$name,function(f){epa_dat[name==f]$e})
V(ferg2)$Betweenness <- betweenness(ferg2) #sapply(V(ferg2)$name,function(f){epa_dat[name==f]$p})
p5 <- ggplot(ferg2, aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_edges( color = "grey50",size=.2,alpha=.5) +geom_nodes(aes(color=e_val),size=3,alpha=1) +
  theme_blank() + scale_color_gradient2("Evaluative", low=muted('red'),
                                        mid='light grey',
                                        high='blue')  + geom_nodetext_repel(aes(label=vertex.names),size=3.5) 

ggsave(file.path(output_dir,"dual_institution.jpeg"),p5,dpi=600,h=4,w=5)
