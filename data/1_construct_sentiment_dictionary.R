library(data.table)
library(ggplot2)
dat <- fread("sentiment_data/Ratings_Warriner_et_al.csv")
epa <- dat[,c("Word","V.Mean.Sum","A.Mean.Sum","D.Mean.Sum"),with=F]
setnames(epa, c("term","e","a","p"))
epa$e <- with(epa, ((e-min(e))*8.6)/(max(e)-min(e)) -4.3)
epa$p <- with(epa, ((p-min(p))*8.6)/(max(p)-min(p)) -4.3)
epa$a <- with(epa, ((a-min(a))*8.6)/(max(a)-min(a)) -4.3)

#looks normal enough
library(fitdistrplus)
plot(fitdist(epa$e, "norm"))

library(readstata13)
library(stringi)
d <- data.table(read.dta13("sentiment_data/FullCleanUGAData.dta"))

d$term <- sub("i_","",d$termID)
d$term <- sub("b_","",d$term)
d$term <- sub("m_","",d$term)
d$term <- stri_replace_all_fixed( d$term,"_", " ")
epa2 <- d[,list(e=mean(E,na.rm=T),p=mean(P,na.rm=T),a=mean(A,na.rm=T)),by="term"]

epa_all <- merge(epa,epa2,by="term")


epa_all_merge <- epa_all[, list(e=(e.x+e.y)/2,p=(p.x+p.y)/2,a=(a.x+a.y)/2),by="term"]
wit_not_in_act <- epa[!(term %in% epa_all$term)]
act_not_in_with <- epa2[!(term %in% epa_all$term)]

all_terms <- rbind(epa_all_merge,wit_not_in_act,act_not_in_with)
write.table(all_terms,"sentiment_data/all_epa_terms.txt",row.names=F,col.names=F,quote=F,sep="\t")