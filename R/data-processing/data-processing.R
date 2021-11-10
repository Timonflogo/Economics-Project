# import libraries
pacman::p_load(dplyr, tidyverse, ggplot2, readxl, rjson, jsonlite)

# import data files

consumption_j <- read.csv("Data/Consumption_kWh.csv")
consumption_t <- read_excel("Data/timon-stuff.xlsx")

test <- rjson::fromJSON(file = "Data/Weather-data.json", flatten = TRUE)
test1 <- fromJSON(txt = "Data/2021/2021-01-01.txt")
test2 <- fromJSON(txt = "Data/2021-05.txt")
