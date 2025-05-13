# Helper functions called by R Templates
# Berk Ustun | www.berkustun.com

required_packages = c('boot', 'base', 'dplyr', 'knitr', 'ggplot2', 'xtable', 'rmarkdown', 'stringr', 'reshape2', 'scales', 'gridExtra', 'grid')
for (pkg in required_packages){
    suppressPackageStartupMessages(library(pkg, character.only = TRUE, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE));
}

#### File Manipulation ####

safe.dir = function(dir_name){
    last_char = substr(dir_name,nchar(dir_name),nchar(dir_name));
    if (last_char != "/") {
        dir_name = paste0(dir_name,"/");
    }
    return(dir_name);
}

open.pdf  = function(pdf_file){
    system(sprintf("open \"%s\"", pdf_file))
}

#### Plot Embellishments ###

default.plot.theme = function(){

    line_color = "#E9E9E9";

    default_theme = theme_bw() +
        theme(title = element_text(size = 18),
              plot.margin = margin(t = 0.25, r = 0, b = 0.75, l = 0.25, unit = "cm"),
              axis.line = element_blank(),
              panel.border = element_rect(size = 2.0, color = line_color),
              panel.grid.minor = element_blank(),
              panel.grid.major = element_line(linetype="solid", size=1.0, color=line_color),
              #
              axis.title.x = element_text(size = 20, margin = margin(t = 20, unit = "pt")),
              axis.text.x   = element_text(size = 20),
              axis.ticks.x  = element_line(size = 1.0, color = line_color),
              #
              axis.title.y = element_text(size = 20, margin = margin(b = 20, unit = "pt")),
              axis.text.y   = element_text(size=20),
              axis.ticks.y	= element_line(size=1.0, color = line_color),
              #
              legend.position="none",
              legend.title = element_blank(),
              legend.text = element_text(face="plain",size=14,angle=0,lineheight=30),
              #legend.key.width = unit(1.5, "cm"),
              #legend.key.height = unit(1.5, "cm"),
              #legend.text.align = 0,
              legend.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "cm"))

    return(default_theme);
}

plot.color.array = function(color_array){

    # More colors here: http://www.color-hex.com/color-palette/2698
    # col = c("#FFBF80", "#FF8000", "#FFFF33","#B2FF8C","#33FF00","#A6EDFF","#1AB2FF","#CCBFFF","#664CFF", "#FF99BF","#E61A33", "#197319","#e3c26c","#887050")
    # image(1:length(col),1,matrix(1:length(col), ncol=1),col=col)

    par(mar=c(10,0,0,0));
    image(x = 1:length(col), y = 1, matrix(1:length(color_array), ncol=1), col=color_array, xlab = "" , ylab ="");
    axis(1, at=seq(1,length(color_array)), labels =names(color_array), las=2, cex=0.5, tick =FALSE);
}

human.numbers = function(x = NULL, smbl =""){
    #https://github.com/fdryan/R/blob/master/ggplot2_formatter.r
    humanity <- function(y){

        if (!is.na(y)){

            b <- round(abs(y) / 1e9, 0.1)
            m <- round(abs(y) / 1e6, 0.1)
            k <- round(abs(y) / 1e3, 0.1)

            if ( y >= 0 ){
                y_is_positive <- ""
            } else {
                y_is_positive <- "-"
            }

            if ( k < 1 ) {
                paste0(y_is_positive, smbl, y )
            } else if ( m < 1){
                paste0 (y_is_positive, smbl,  k , "K")
            } else if (b < 1){
                paste0 (y_is_positive, smbl, m ,"M")
            } else {
                paste0 (y_is_positive, smbl,  comma(b), "N")
            }
        }
    }
    sapply(x,humanity)
}

label.digits = function(l) {
    # turn in to character string in scientific notation
    na_ind = which(is.na(l))
    l <- format(l, nsmall = 3, scientific = FALSE);
    l[na_ind] = NA
    return(l)
}

label.auc = function(l) {
    # turn in to character string in scientific notation
    na_ind = which(is.na(l))
    l <- format(l, nsmall = 3, scientific = FALSE);
    l[na_ind] = NA
    return(l)
}

grab.legend = function(p){
    tmp = ggplot_gtable(ggplot_build(p))
    leg = which(sapply(tmp$grobs, function(y) y$name) == "guide-box")
    legend_plot = tmp$grobs[[leg]]
    return(legend_plot)
}

#### Table Creation ####

sanitize.tex = function(x){
    sanitize(x, type = "latex")
}

bold.tex = function(x){
    paste0('{\\bfseries ', x, '}')
}

print.bfcell.header = function(header){
    header = gsub("\\","\\\\",header,fixed=TRUE)
    header = sprintf("\\bfcell{c}{%s}",header);
    return(header)
}

print.tex.header = function(header){
    header = gsub("\\","\\\\",header,fixed=TRUE)
    header = sprintf("\\begin{tabular}{>{\\bf}c}%s\\end{tabular}",header);
    return(header)
}
