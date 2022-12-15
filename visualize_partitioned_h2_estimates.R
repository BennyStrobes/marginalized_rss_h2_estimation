args = commandArgs(trailingOnly=TRUE)
library(cowplot)
library(ggplot2)
library(hash)
library(dplyr)
library(reshape)
library(stringr)
options(warn=1)

figure_theme <- function() {
	return(theme(plot.title = element_text(face="plain",size=11), text = element_text(size=11),axis.text=element_text(size=11), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black"), legend.text = element_text(size=11), legend.title = element_text(size=11)))
}


load_in_ldsc_taus <- function(trait_name, ldsc_results, ldsc_evd_results, standard_sldsc_results) {
	anno_names_arr <- c()
	tau_arr <- c()
	tau_se_arr <- c()
	model_name_arr <- c()

	# Standard sldsc
	standard_ldsc_file <- paste0(standard_sldsc_results, trait_name, "_sldsc_res_.results")
	standard_ldsc <- read.table(standard_ldsc_file, header=TRUE, sep="\t")
	anno_names <- as.character(standard_ldsc$Category)

	anno_names_arr <- c(anno_names_arr, anno_names)
	tau_arr <- c(tau_arr, standard_ldsc$Coefficient)
	tau_se_arr <- c(tau_se_arr, standard_ldsc$Coefficient_std_error)
	model_name_arr <- c(model_name_arr, rep("standard_sldsc", length(anno_names)))


	# Block ldsc
	block_ldsc_file <- paste0(ldsc_results, trait_name, "_10_mb_windows_ldsc_results_tau_bootstrapped_ci.txt")
	block_ldsc <- read.table(block_ldsc_file, header=TRUE, sep="\t")

	anno_names_arr <- c(anno_names_arr, anno_names)
	tau_arr <- c(tau_arr, block_ldsc$estimate)
	tau_se_arr <- c(tau_se_arr, block_ldsc$standard_error)
	model_name_arr <- c(model_name_arr, rep("block_sldsc", length(anno_names)))	

	# evd ldsc
	evd_ldsc_file <- paste0(ldsc_evd_results, trait_name, "_10_mb_windows_ldsc_results_tau_bootstrapped_ci.txt")
	evd_ldsc <- read.table(evd_ldsc_file, header=TRUE, sep="\t")

	anno_names_arr <- c(anno_names_arr, anno_names)
	tau_arr <- c(tau_arr, evd_ldsc$estimate)
	tau_se_arr <- c(tau_se_arr, evd_ldsc$standard_error)
	model_name_arr <- c(model_name_arr, rep("evd_sldsc", length(anno_names)))	


	df <- data.frame(annotation=anno_names_arr, tau=tau_arr, tau_se=tau_se_arr, model=model_name_arr)


	return(df)

}

make_boxplot_comparing_standard_error_sizes <- function(ldsc_tau_df, model_name1, model_name2) {
	standard_error <- c()
	model_name <- c()

	model1_se = ldsc_tau_df[ldsc_tau_df$model==model_name1,]$tau_se
	model2_se = ldsc_tau_df[ldsc_tau_df$model==model_name2,]$tau_se

	print(wilcox.test(model1_se, model2_se, paired=TRUE,alternative = "two.sided"))

	df <- data.frame(standard_error=c(model1_se, model2_se), model_name=c(rep(model_name1, length(model1_se)), rep(model_name2, length(model2_se))))

	p <- ggplot(df, aes(x=model_name, y=standard_error)) + 
  			geom_boxplot() +
  			figure_theme()
  	return(p)
}

make_histo_comparing_standard_error_sizes <- function(ldsc_tau_df, model_name1, model_name2) {
	standard_error <- c()
	model_name <- c()

	model1_se = ldsc_tau_df[ldsc_tau_df$model==model_name1,]$tau_se
	model2_se = ldsc_tau_df[ldsc_tau_df$model==model_name2,]$tau_se

	diff = model1_se - model2_se

	print(wilcox.test(model1_se, model2_se, paired=TRUE,alternative = "two.sided"))

	df <- data.frame(diff=diff)

	p <- ggplot(df, aes(x=diff)) + 
  			geom_histogram() +
  			figure_theme() + 
  			labs(x=paste0(model_name1, " SE - ", model_name2, " SE"), y="Num annotations")
  	return(p)
}



make_scatter_plot_comparing_taus <- function(df, model_1_name, model_2_name) {
	anno_arr <- c()
	model1_tau_arr <- c()
	model2_tau_arr<- c()
	model1_max_arr <- c()
	model1_min_arr <- c()
	model2_max_arr <- c()
	model2_min_arr <- c()


	annotations <- unique(df$annotation)

	for (anno_iter in 1:length(annotations)) {
		anno_name <- annotations[anno_iter]
		anno_df <- df[df$annotation==anno_name, ]
		anno_model1_df <- anno_df[anno_df$model==model_1_name, ]
		anno_model2_df <- anno_df[anno_df$model==model_2_name, ]

		model1_tau <- anno_model1_df$tau
		model1_tau_se <- anno_model1_df$tau_se
		model2_tau <- anno_model2_df$tau
		model2_tau_se <- anno_model2_df$tau_se
		model_1_max <- model1_tau + 1.96*model1_tau_se
		model_1_min <- model1_tau - 1.96*model1_tau_se
		model_2_max <- model1_tau + 1.96*model2_tau_se
		model_2_min <- model1_tau - 1.96*model2_tau_se

		anno_arr <- c(anno_arr, anno_name)
		model1_tau_arr <- c(model1_tau_arr, model1_tau)
		model2_tau_arr <- c(model2_tau_arr, model2_tau)
		model1_max_arr <- c(model1_max_arr, model_1_max)
		model2_max_arr <- c(model2_max_arr, model_2_max)
		model1_min_arr <- c(model1_min_arr, model_1_min)
		model2_min_arr <- c(model2_min_arr, model_2_min)
	}

	df2<- data.frame(annotation=anno_arr, model1_tau=model1_tau_arr, model2_tau=model2_tau_arr, model1_min=model1_min_arr, model1_max=model1_max_arr,model2_min=model2_min_arr, model2_max=model2_max_arr)

	p <- ggplot(data = df2,aes(x = model1_tau,y = model2_tau)) + 
    	geom_point() + 
    	geom_errorbar(aes(ymin = model2_min,ymax = model2_max)) + 
   	 	geom_errorbarh(aes(xmin = model1_min,xmax = model1_max)) +
   	 	figure_theme() +
   	 	geom_abline(color='grey') + 
   	 	labs(x=paste0(model_1_name, " tau"),y=paste0(model_2_name, " tau"))
   	 return(p)
}



######################
# Command line args
######################
ldsc_results=args[1]
ldsc_evd_results=args[2]
standard_sldsc_results=args[3]
h2_viz_dir=args[4]


trait_name <- "UKB_460K.blood_WHITE_COUNT"
ldsc_tau_df <- load_in_ldsc_taus(trait_name, ldsc_results, ldsc_evd_results, standard_sldsc_results)



scatter <- make_scatter_plot_comparing_taus(ldsc_tau_df, "standard_sldsc", "block_sldsc")
output_file <- paste0(h2_viz_dir , "standard_ldsc_vs_block_ldsc_tau_scatter.pdf")
ggsave(scatter, file=output_file, width=7.2, height=5.0, units="in")


boxplot <- make_boxplot_comparing_standard_error_sizes(ldsc_tau_df, "standard_sldsc", "block_sldsc")
output_file <- paste0(h2_viz_dir , "standard_ldsc_vs_block_ldsc_tau_se_boxplot.pdf")
ggsave(boxplot, file=output_file, width=7.2, height=5.0, units="in")


histogram <- make_histo_comparing_standard_error_sizes(ldsc_tau_df, "block_sldsc", "standard_sldsc")
output_file <- paste0(h2_viz_dir , "standard_ldsc_vs_block_ldsc_tau_se_histo.pdf")
ggsave(histogram, file=output_file, width=7.2, height=4.0, units="in")


scatter <- make_scatter_plot_comparing_taus(ldsc_tau_df, "block_sldsc", "evd_sldsc")
output_file <- paste0(h2_viz_dir , "evd_ldsc_vs_block_ldsc_tau_scatter.pdf")
ggsave(scatter, file=output_file, width=7.2, height=5.0, units="in")

