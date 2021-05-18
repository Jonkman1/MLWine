# Harrie Jonkman
# MLRegression on Wine data
# 18 May

# loading packages
library(tidyverse)
library(finalfit)
library(tidymodels)
library(caret)
library(rpart)   
library(randomForest)

# DATA LOADING AND PREPROCESSING 
wf<-readRDS("wine.rds")
                  
# Let us give a summary of the data frame.
glimpse(wf)

# Do we see missings?
missing_glimpse(wf)
                    
# What kind of variables do we have and what are their scores?
ff_glimpse(wf)
                    
# Let us make correlation matrix of the data.
cor(wf)

# Let us define the column names on a consistent way.
colnames(wf) <- wf %>% 
        colnames() %>% str_replace_all(pattern = " ", replacement = "_")
colnames(wf)

# Let us also remove any missing values.
wf <- na.omit(wf)
                    
## EXPLORATIVE DATA ANALYSIS (EDA).
library(corrplot)
wf %>% cor() %>% 
       corrplot.mixed(upper = "circle",
       tl.cex = 1,
       tl.pos = 'lt',
       number.cex = 0.75)
                    
## SPLITTING THE DATA
set.seed(12345) 

data_split <- initial_split(wf, prop = 0.8) 

train_data <- training(data_split)

test_data  <- testing(data_split)

# MODELING AND DATA ANALYSIS 

# 1. Lineair modelling
formula <- formula(quality ~ fixed_acidity + volatile_acidity + citric_acid + 
                     residual_sugar + chlorides + free_sulfur_dioxide + 
                     total_sulfur_dioxide + density + pH + sulphates + alcohol)

lm_fit <- 
  linear_reg() %>% 
  set_mode("regression") %>% 
  set_engine("lm") %>% 
  fit(formula, data = train_data) 

# Show the results
print(lm_fit$fit)

# Show the results in another way.
summary(lm_fit$fit)

# Or on broom-way
tidy(lm_fit$fit) %>% mutate_if(is.numeric, round, 3)

# 2. Decision tree

dt_fit <- 
  decision_tree() %>% 
  set_mode("regression") %>%
  set_engine("rpart") %>% 
  fit(formula, data = train_data)

#cPrint the dt-results
print(dt_fit$fit)


# As a sidestep we can visualise this
library(visNetwork)
library(sparkline)
visTree(dt_fit$fit)

## 3. Random forest
rf_fit <- 
  rand_forest() %>% 
  set_mode("regression") %>%
  set_engine("randomForest") %>% 
  fit(formula, data = train_data)

# Print these results.
print(rf_fit$fit)

# EVALUATION AND PREDICTION
#  We compare three models (`lm_fit`, `dt_fit` and `rf_fit`) on the MSE score.
# Accuracy of the lm-model
lm_pred <- test_data %>% 
  bind_cols(predict(object = lm_fit, new_data = test_data))

# Now we see a new column, `.pred`, with a predicted scores for each row. 
View(lm_pred)

lm_pred <- test_data %>% 
  bind_cols(predict(object = lm_fit, new_data = test_data)) %>% 
  mutate(pred = round(.pred, 0))


lm_mse <- lm_pred %>% 
  summarise(type = "lm",
            MSE = round(mean((pred - quality)^2), 4))

head(lm_mse)

# Accuracy of the Decision Tree Model
dt_pred <- test_data %>% 
  bind_cols(predict(object = dt_fit, new_data = test_data)) %>% 
  rename(pred = .pred) %>% 
  mutate(pred = round(pred, 0))

dt_mse <- dt_pred %>% 
  summarise(type = "dt",
            MSE = round(mean((pred - quality)^2), 4))

head(dt_mse)

## Accuracy of the Random Forest Model
rf_pred <- test_data %>% 
  bind_cols(predict(object = rf_fit, new_data = test_data)) %>% 
  rename(pred = .pred) %>% 
  mutate(pred = round(pred, 0))

rf_mse <- rf_pred %>% 
  summarise(type = "rf",
            MSE = round(mean((pred - quality)^2), 4))
head(rf_mse)

## All results together
res <- bind_rows(lm_mse, dt_mse, rf_mse)

# Let us show these results.
head(res)

# We choose the random_forest model as the best opportunity here. Let us look at it once again.
View(rf_pred)

head(rf_pred)

# RESULTS SUMMARIZED
head(rf_pred)

metrics(rf_pred, quality, pred)


