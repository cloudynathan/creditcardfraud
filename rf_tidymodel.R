# ------ ranger tidymodel
library(tidyverse)
library(tidymodels)
library(workflows)
library(tune)

#load data
df <- read.csv("C:/workspaceR/creditcardfraud/creditcard.csv")

#check for NAs and convert Class to factor
anyNA(df)

df <- df %>% mutate(Class = as_factor(Class))

#set seed and split data into training and testing
set.seed(123)
df_split <- initial_split(df)
df_train <- training(df_split)
df_test <- testing(df_split)

#in the training and testing datasets, how many are fraudulent transactions?
df_train %>% count(Class)
df_test %>% count(Class)

#ranger model
model_rf <- 
  rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

#grid of hyperparameters
grid_rf <- 
  grid_max_entropy(        
    mtry(range = c(1, 20)), 
    trees(range = c(500, 1000)),
    min_n(range = c(2, 10)),
    size = 30)

#workflows
wkfl_rf <- 
  workflow() %>% 
  add_formula(Class ~ .) %>% 
  add_model(model_rf)

#cross validation method
cv_folds <- vfold_cv(df_train, v = 5)
cv_folds

#choose metrics
my_metrics <- metric_set(roc_auc, accuracy, sens, spec, precision, recall)

#tuning
rf_fit <- tune_grid(
  wkfl_rf,
  resamples = cv_folds,
  grid = grid_rf,
  metrics = my_metrics,
  control = control_grid(verbose = TRUE)
)

#inspect tuning 
rf_fit
collect_metrics(rf_fit)
autoplot(rf_fit, metric = "roc_auc")
show_best(rf_fit, metric = "roc_auc", maximize = TRUE)
select_best(rf_fit, metric = "roc_auc", maximize = TRUE)

# Fit best model 
tuned_model <-
  wkfl_rf %>% 
  finalize_workflow(select_best(rf_fit, metric = "roc_auc", maximize = TRUE)) %>% 
  fit(data = df_train)

predict(tuned_model, df_train)
predict(tuned_model, df_test)
rf_predictions <- predict(tuned_model, df_test) %>% bind_cols(df_test) %>% select(.pred_class, Class)



