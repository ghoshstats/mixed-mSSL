library(tidygraph)
library(ggraph)
library(dplyr)
library(igraph)   

species_data <- read.csv("~/hmsc_data.csv")
da = droplevels(subset(species_data, Year %in% 2014))
prev_da = droplevels(subset(species_data, Year %in% 2013))
XData = data.frame(clim = da$AprMay, effort=da$Effort, junjul=da$JunJul,djf=da$DJF)
med_Y <- apply(prev_da[, 10:59], 
               MARGIN = 2,           
               FUN    = median,      
               na.rm  = TRUE) 

Y <- as.matrix((da[, 10:59] > med_Y) * 1)
min_max_scaling <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

XData <- apply(XData, 2, min_max_scaling)

mixed_model <- mixed_mssl(XData,Y,response_types = rep("binary",50),lambdas = list(lambda1 = 0.001, lambda0 = 5),
                          xis = list(xi1 = 0.00001 * nrow(XData), xi0 = seq(0.0001 * nrow(XData), nrow(XData), length = 10)),
                          theta_hyper_params = c(1, ncol(XData)*ncol(Y)),
                          eta_hyper_params = c(1, ncol(Y)),
                          diag_penalty = 0,
                          max_iter = 500,
                          eps = 1e-3,
                          s_max_condition = 10 * nrow(XData),
                          obj_counter_max = 5,
                          verbose = 1, nrep = 1000, nskp = 1)

Omega_matrix <- mixed_model$Omega

Omega_matrix[Omega_matrix > 0] <- Omega_matrix[Omega_matrix > 0] * 2
adj_mat <- as.matrix(Omega_matrix)
diag(adj_mat) <- 0
thr <- quantile(abs(adj_mat[adj_mat!=0]), .90)    # top 5% strongest edges
g2 <- graph_from_adjacency_matrix(adj_mat, 
                                  mode="undirected", weighted=TRUE, diag=FALSE) %>%
  delete_edges(E(.)[ abs(weight) < thr ]) %>%
  delete_vertices(degree(.)==0)

tg <- as_tbl_graph(g2)

set.seed(2)
#tiff("Partial-Correlation.tiff",units="in",width=10,height=6,res=600)
ggraph(tg, layout = "in_circle") + 
  geom_edge_arc(aes(
    edge_colour = weight > 0,
    edge_width  = abs(weight)
  ),
  strength = 0.01,       
  alpha    = 0.8
  ) +
  scale_edge_colour_manual("", values=c("TRUE"="black","FALSE"="red"),
                           labels=c("+","−")) +
  scale_edge_width_continuous(range = c(0.5, 2)) +
  
  # nodes: simple circles
  geom_node_point(size = 4, color = "darkgreen") +
  geom_node_text(aes(label = name),
                 repel    = TRUE,
                 size     = 3,
                 family   = "sans",
                 color    = "black",
                 point.padding = unit(0.2, "lines")
  ) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust=0.5)
  ) +
  labs(
    title = "Top 10% Strongest Partial–Correlation Network",
    edge_width = "Strength"
  )

dev.off()

