library(groupdata2)
library(tidyverse)
library(tools)
library(utils)


relative_path <- file.path('data', 'images')

all_data <- list.files(relative_path) %>%
    as.data.frame() %>%
    rename(filepath = '.') %>%
    separate(filepath, c('annotation', 'studyinstanceuid', 'image_number', 'ext'), remove = FALSE) %>%
    mutate(across(c('studyinstanceuid', 'annotation'), as.factor),
           filepath = file.path(relative_path, filepath)) %>%
    select(filepath, studyinstanceuid, annotation)


set.seed(1337)

partitioned_data <- all_data %>%
    partition(p = c(0.7, 0.1),
              id_col = 'studyinstanceuid')

train_data <- partitioned_data[[1]] %>%
    select(filepath, studyinstanceuid, annotation)

val_data <- partitioned_data[[2]] %>%
    select(filepath, studyinstanceuid, annotation)

test_data <- partitioned_data[[3]] %>%
    select(filepath, studyinstanceuid, annotation)

write_csv(train_data, './data/training.csv')
write_csv(val_data, './data/validation.csv')
write_csv(test_data, './data/testing.csv')
