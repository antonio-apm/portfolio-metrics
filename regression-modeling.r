
#install.packages('tidyverse')
library(tidyverse)

# import data
macro = read_csv('macro.csv')
returns = read_csv('returns.csv')

returns = returns %>% rename(Return = `0`)

# removing first row of NA observations
macro = macro[2:nrow(macro), ] 
returns = returns[2:nrow(returns), ]

df = cbind(macro, returns)

for (pred in names(macro)) {
    if (pred != "Date") {
        plot(df[[pred]], df[['Return']], main=paste0("Returns (y) vs. ", pred, " (x)"))
    }
}

## ok just do the rest in Python...