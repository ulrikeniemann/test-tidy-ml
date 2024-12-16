# ..............................................................................
#
# Export Daten für ML-Tests
#
# langes Datenformat
# abh. Var: VZÄ(?)
#
# ..............................................................................


# csv
#write_csv2(w, file = "test-ml-dash-bua.csv", na = "")
w <- read_csv2(file = "test-ml-dash-bua.csv") %>% as_tibble()

# ..............................................................................

# Chapter 2

# 2.1.2 Reuse existing data structures
boot_samp <- rsample::bootstraps(w, times = 3)
boot_samp

# Chapter 3
ggplot(w, 
       aes(x = ma, y = `1010200001`, 
           color = RaumID, pch = RaumID)) + 
  # Plot points for each data point and color by RaumID
  geom_point(size = 2) + 
  # Show a simple linear model fit created separately for each RaumID:
  geom_smooth(method = lm, se = FALSE, alpha = 0.5) + 
  scale_color_brewer(palette = "Paired") +
  labs(x = "Mitarbeitende", y = "Kund:innen")

# To fit an ordinary linear model in R, the lm() function is commonly used.
interaction_fit <-  lm(ma ~ `1010200001` + `1030800001`, data = w) 

# To print a short summary of the model:
interaction_fit

# Place two plots next to one another:
par(mfrow = c(1, 2))

# Show residuals vs predicted values:
plot(interaction_fit, which = 1)

# A normal quantile plot on the residuals:
plot(interaction_fit, which = 2)

# Fit a reduced model:
main_effect_fit <-  lm(ma ~ `1010200001`, data = w) 

# Compare the two:
anova(main_effect_fit, interaction_fit)

summary(main_effect_fit)


corr_res <- map(w %>% select(-ma, -RaumID, -Datum), 
                cor.test, y = w$ma)
corr_res[[1]]
library(broom)
tidy(corr_res[[1]])

corr_res %>% 
  # Convert each to a tidy format; `map_dfr()` stacks the data frames 
  map_dfr(tidy, .id = "predictor") %>% 
  ggplot(aes(x = fct_reorder(predictor, estimate))) + 
  geom_point(aes(y = estimate)) + 
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = .1) +
  labs(x = NULL, y = "Correlation with ma") +
  # Rotating axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# 3.4 Combining Base R Models and the Tidyverse
split_by_species <- 
  w %>% 
  group_nest(RaumID) 
split_by_species

model_by_species <- 
  split_by_species %>% 
  mutate(model = map(data, ~ lm(ma ~ `1010100012`, data = .x)))
model_by_species

model_by_species %>% 
  mutate(coef = map(model, tidy)) %>% 
  select(RaumID, coef) %>% 
  unnest(cols = c(coef))

# 3.5 The tidymodels Metapackage
library(tidymodels)
library(conflicted)
conflict_prefer("filter", winner = "dplyr")
tidymodels_prefer(quiet = FALSE)

# ..............................................................................
# Modeling Basics
# ..............................................................................

# Chapter 4

dim(w)

ggplot(w, aes(x = ma)) + 
  geom_histogram(bins = 50, col= "white")

# When modeling this outcome, a strong argument can be made that the price 
# should be log-transformed. The advantages of this type of transformation 
# are that no houses would be predicted with negative sale prices and that 
# errors in predicting expensive houses will not have an undue influence 
# on the model. Also, from a statistical perspective, a logarithmic transform 
# may also stabilize the variance in a way that makes inference more legitimate. 
# We can use similar steps to now visualize the transformed data
ggplot(w, aes(x = ma)) + 
  geom_histogram(bins = 50, col= "white") +
  scale_x_log10()
# The disadvantages of transforming the outcome mostly relate 
# to interpretation of model results.

# test to log (?)
w <- w %>% mutate(maLog = log10(ma))

# ..............................................................................

# Chapter 5: Spending our Data

# Set the random number stream using `set.seed()` so that the results can be 
# reproduced later. 
set.seed(501)

# Save the split information for an 80/20 split of the data
w_split <- initial_split(w, prop = 0.80)
w_split

w_train <- training(w_split)
w_test  <-  testing(w_split)

dim(w_train)

# A stratified random sample would conduct the 80/20 split within each of 
# these data subsets and then pool the results.
set.seed(502)
w_split <- initial_split(w, prop = 0.80, strata = ma)
w_train <- training(w_split)
w_test  <-  testing(w_split)

dim(w_train)

# 5.2 What About a Validation Set?
set.seed(52)
# To put 60% into training, 20% in validation, and 20% in testing:
w_val_split <- initial_validation_split(w, prop = c(0.6, 0.2))
w_val_split

# To get the training, validation, and testing data, the same syntax is used:
w_train <- training(w_val_split)
w_test <- testing(w_val_split)
w_val <- validation(w_val_split)

# Data splitting is the fundamental strategy for empirical validation of models.
# At this checkpoint, the important code snippets for preparing and splitting are:
# library(tidymodels)
# data(ames)
# ames <- ames %>% mutate(Sale_Price = log10(Sale_Price))
# 
# set.seed(502)
# ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
# ames_train <- training(ames_split)
# ames_test  <-  testing(ames_split)

# ..............................................................................

# Chapter 6: Fitting Models with parsnip

# A variety of methods can be used to estimate the model parameters:
# Ordinary linear regression uses the traditional method of least squares 
# to solve for the model parameters.
# Regularized linear regression adds a penalty to the least squares method 
# to encourage simplicity by removing predictors and/or shrinking their 
# coefficients towards zero. This can be executed using Bayesian or 
# non-Bayesian techniques.

# model <- lm(formula, data, ...)
# model <- stan_glm(formula, data, family = "gaussian", ...)
# model <- glmnet(x = matrix, y = vector, family = "gaussian", ...)

# For tidymodels, the approach to specifying a model is intended to be more unified

# 1) Specify the type of model based on its mathematical structure 
# (e.g., linear regression, random forest, KNN, etc).
# 2) Specify the engine for fitting the model. 
# Most often this reflects the software package that should be used, 
# like Stan or glmnet. These are models in their own right, and parsnip provides 
# consistent interfaces by using these as engines for modeling.
# 3) When required, declare the mode of the model. 
# The mode reflects the type of prediction outcome. 
# For numeric outcomes, the mode is regression; for qualitative outcomes, 
# it is classification.14 If a model algorithm can only address one type of 
# prediction outcome, such as linear regression, the mode is already set.

library(tidymodels)
tidymodels_prefer()

# for example, for the three cases we outlined
linear_reg() %>% set_engine("lm")
linear_reg() %>% set_engine("glmnet") 
linear_reg() %>% set_engine("stan")

# The translate() function can provide details on how parsnip converts 
# the user’s code to the package’s syntax

linear_reg() %>% set_engine("lm") %>% translate()
linear_reg(penalty = 1) %>% set_engine("glmnet") %>% translate()
linear_reg() %>% set_engine("stan") %>% translate()


lm_model <- 
  linear_reg() %>% 
  set_engine("lm")

lm_form_fit <- 
  lm_model %>% 
  # Recall that Sale_Price has been pre-logged
  #fit(Sale_Price ~ Longitude + Latitude, data = ames_train)
  fit(ma ~ Datum + `1010100012`, data = w_train)

lm_xy_fit <- 
  lm_model %>% 
  fit_xy(
    x = w_train %>% select(Datum, `1010100012`),
    y = w_train %>% pull(ma)
  )

lm_form_fit
lm_xy_fit

# To understand how the parsnip argument names map to the original names, 
# use the help file for the model (available via ?rand_forest) as well 
# as the translate() function:
rand_forest(trees = 1000, min_n = 5) %>% 
  set_engine("ranger") %>% 
  set_mode("regression") %>% 
  translate()

# Modeling functions in parsnip separate model arguments into two categories:
# Main arguments are more commonly used and tend to be available across engines.
# Engine arguments are either specific to a particular engine or used more rarely.

# For example, to have the ranger::ranger() function print out more information about the fit:
rand_forest(trees = 1000, min_n = 5) %>% 
  set_engine("ranger", verbose = TRUE) %>% 
  set_mode("regression") 

# 6.2 Use the Model Results

lm_form_fit %>% extract_fit_engine()
lm_form_fit %>% extract_fit_engine() %>% vcov()

model_res <- 
  lm_form_fit %>% 
  extract_fit_engine() %>% 
  summary()
# The model coefficient table is accessible via the `coef` method.
param_est <- coef(model_res)
class(param_est)
param_est

tidy(lm_form_fit)

# 6.3 Make Predictions

# For predictions, parsnip always conforms to the following rules:
# 1) The results are always a tibble.
# 2) The column names of the tibble are always predictable.
# 3) There are always as many rows in the tibble as there are in the input data set.

w_test_small <- w_test %>% slice(1:5)
predict(lm_form_fit, new_data = w_test_small)

# merge predictions with the original data

w_test_small %>% 
  select(ma) %>% 
  bind_cols(predict(lm_form_fit, w_test_small)) %>% 
  # Add 95% prediction intervals to the results:
  bind_cols(predict(lm_form_fit, w_test_small, type = "pred_int")) 

tree_model <- 
  decision_tree(min_n = 2) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")
tree_fit <- 
  tree_model %>% 
  fit(ma ~ Datum + `1010100012`, data = w_train)
w_test_small %>% 
  select(ma) %>% 
  bind_cols(predict(tree_fit, w_test_small))

# 6.4 parsnip-Extension Packages
# https://www.tidymodels.org/find/

# 6.5 Creating Model Specifications

# It may become tedious to write many model specifications, or to remember 
# how to write the code to generate them. The parsnip package includes an 
# RStudio addin16 that can help. Either choosing this addin from the 
# Addins toolbar menu or running the code
parsnip_addin()

# 6.6 Chapter Summary

# The code for modeling the Ames data that we will use moving forward is:
#   
# library(tidymodels)
# data(ames)
# ames <- mutate(ames, Sale_Price = log10(Sale_Price))
# 
# set.seed(502)
# ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
# ames_train <- training(ames_split)
# ames_test  <-  testing(ames_split)
# 
# lm_model <- linear_reg() %>% set_engine("lm")

# ..............................................................................

