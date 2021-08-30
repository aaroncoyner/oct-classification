library(groupdata2)
library(tidyverse)
library(tools)



all_data <- list.files('./data/images') %>%
    as.data.frame() %>%
    rename(filepath = '.') %>%
    separate(filepath, c('annotation', 'studyinstanceuid', 'image_number', 'ext'), remove = FALSE) %>%
    mutate(across(c('studyinstanceuid', 'annotation'), as.factor)) %>%
    select(filepath, studyinstanceuid, annotation)


set.seed(1337)

partitioned_data <- all_data %>%
    partition(p = c(0.7, 0.1),
              id_col = 'studyinstanceuid')

train_data <- partitioned_data[[1]]
val_data <- partitioned_data[[2]]
test_data <- partitioned_data[[3]]

write_csv(train_data, './data/training.csv')
write_csv(val_data, './data/validation.csv')
write_csv(test_data, './data/testing.csv')
