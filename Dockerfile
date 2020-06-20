FROM jupyter/datascience-notebook:dc9744740e12

RUN pip install stanza==1.0.1\
                torch==1.5.0\
                transformers==2.9.0 &&\
    R -e "install.packages(c('tidyverse', 'lme4', 'ggrepel'), repos='http://cran.us.r-project.org')"
