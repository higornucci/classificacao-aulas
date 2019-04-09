install.packages(readr)
install.packages(qplot)
install.packages(ggplot2)
library(readr)
library(ggplot2)
library(gplots)

dadosClassificador<-read.csv("/home/ufms/projetos/classificacao-aulas/dissertacao/r-studio/f1_scores.csv")

dadosClassificador.anova<-aov(dadosClassificador$metrica ~ dadosClassificador$modelo + dadosClassificador$classe)
# Mostra a tabela ANOVA
summary(dadosClassificador.anova)


#----------------------------------------------------------------------------------------------------
# Realiza e mostra os resultados de um pos-teste usando Tukey
TukeyHSD(dadosClassificador.anova, conf.level = 0.95, ordered = TRUE)
plot(TukeyHSD(dadosClassificador.anova, conf.level = 0.95),las=1, col = "blue")
plot(dadosClassificador.anova)

#----------------------------------------------------------------------------------------------------
#Métricas Desempenho dos Classificadores
ggplot(dadosClassificador, 
       aes(x=modelo,
           y=metrica, 
           fill=modelo
       )) +
  labs(fill = "Classifiers") +
  theme(legend.position="none") +
  xlab("Classifiers") +
  ylab("Performance Metrics")+
  #ylab("Métricas de desempenho")+
  #xlab("Classificadores") +
  geom_boxplot(outlier.size=4, outlier.colour='red',  outlier.shape = 8,  alpha = 0.5) + #outlier e caixa
  geom_dotplot(binaxis='y', stackdir='center',  dotsize=1.2, binwidth=0.4)  # mostra os pontos com métricas

