# ------ xgboost tidymodel
library(tidyverse)
library(tidymodels)
library(workflows)
library(tune)
library(xgboost)

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

#xgboost model
model_xgboost <- 
  boost_tree(mtry = tune(), trees = tune(), min_n = tune()) %>% 
  set_engine("xgboost", importance = "impurity") %>% 
  set_mode("classification")

#grid of hyperparameters
grid_xgboost <- 
  grid_max_entropy(        
    mtry(range = c(1, 20)), 
    trees(range = c(500, 1000)),
    min_n(range = c(2, 10)),
    size = 30)

#workflows
wkfl_xgboost <- 
  workflow() %>% 
  add_formula(Class ~ .) %>% 
  add_model(model_xgboost)

#cross validation method
cv_folds <- vfold_cv(df_train, v = 5)
cv_folds

#choose metrics
my_metrics <- metric_set(roc_auc, accuracy, sens, spec, precision, recall)

#tuning
xgboost_fit <- tune_grid(
  wkfl_xgboost,
  resamples = cv_folds,
  grid = grid_xgboost,
  metrics = my_metrics,
  control = control_grid(verbose = TRUE)
)

#inspect tuning 
xgboost_fit
collect_metrics(xgboost_fit)
autoplot(xgboost_fit, metric = "roc_auc")
show_best(xgboost_fit, metric = "roc_auc", maximize = TRUE)
select_best(xgboost_fit, metric = "roc_auc", maximize = TRUE)

#fit best model 
tuned_model <-
  wkfl_xgboost %>% 
  finalize_workflow(select_best(xgboost_fit, metric = "roc_auc", maximize = TRUE)) %>% 
  fit(data = df_train)

predict(tuned_model, df_train)
predict(tuned_model, df_test)

xgboost_predictions <- 
  predict(tuned_model, df_test) %>% bind_cols(df_test) %>% select(.pred_class, Class)



