# "Machine learning on Wine data. CLASSIFICATION"
# Harrie Jonkman
# date: 18 May 2021
# output: R Script
  
# Downloading packages
library(tidyverse)
library(tidymodels)
library(caret)
library(rpart)   
library(randomForest)
library(visdat)
library(GGally)
library(gt)
library(skimr)
library(corrplot)
library(visNetwork)
library(sparkline)



# INTRODUCTION


# PROBLEM DEFINITION

# DATA LOADING AND PREPROCESSING 
wf<-readRDS("wine.rds")

glimpse(wf)

# To get a first impression of the data we take a look at the top 4 rows:
library(gt)
wf %>% 
slice_head(n = 4) %>% 
gt() # print output using gt

# Next, we take a look at the data structure and check wether all data formats are correct:
glimpse(wf)
                    
# The package `visdat` helps us to explore the data class structure visually:
library(visdat)
vis_dat(wf)

# Missing data
vis_miss(wf, sort_miss = TRUE)

# Create new variables
percentage <- prop.table(table(wf$quality)) * 100
cbind(freq=table(wf$quality), percentage=percentage)

# With `recipe` package we can do a lot at the same time
set.seed(123)
wdNA<-
wf %>%
mutate(
# Convert quality to a factor (for classification)
quality = ifelse(quality >5, "high", "low"),
quality = factor(quality)
      )

# Take out missing data.
wdNA<-na.omit(wdNA)

# Check the missings
vis_miss(wdNA, sort_miss = TRUE)

# Let us look at the dependent variable closely.
percentage <- prop.table(table(wdNA$quality)) * 100
cbind(freq=table(wdNA$quality), percentage=percentage)

# Let us plot it
wdNA %>%
ggplot(aes(quality)) +
geom_bar() +
ggtitle("Quality of wine, Low (0) and High (1)")

# Fix column names
                    
colnames(wdNA) <- wdNA %>% 
            colnames() %>% str_replace_all(pattern = " ", replacement = "_")
colnames(wdNA)
                    
# Data overview
library(skimr)
skim(wdNA)

                    
#We have                      -
library(GGally)
wdNA %>%
ggscatmat(alpha=0.2)
                    ```
# Another option is to use ggpairs:
                      
wdNA %>%
ggpairs()

# A summary of the data frame before the next step
summary(wdNA)
glimpse(wdNA)
                    
## EXPLORATIVE DATA ANALYSIS (EDA).
library(corrplot)
            wdNA %>%
select(-quality) %>%
           cor() %>% 
           corrplot.mixed(upper = "circle",
                                tl.cex = 1,
                             tl.pos = 'lt',
                         number.cex = 0.75)
                    
## Split the data into: a) Train set, b) Test set
set.seed(12345) # to fix randomisation by setting the seed (reproducibility)
                    
data_split <- initial_split(wdNA, prop = 0.8) # Use 80% of the data for training
                    
train_data <- training(data_split)
                    
test_data  <- testing(data_split)

## Validation set
set.seed(100)
wine_cv <-
        vfold_cv(train_data, 
        v = 5, 
        strata = quality) 
                    ```
# MODELING AND DATA ANALYSIS 
## Prepare models
wdNA_rec <-
        recipe(quality ~ .,
        data=train_data) %>%
        prep()
wdNA_rec

## Execute pre-processing
wdNATEST <-wdNA_rec %>%
bake(testing(data_split))
glimpse(wdNATEST)

# Train data has already been prepped. To load the prepared training data into a variable, we use juice(). 
# It will extract the data from the iris_recipe object.
wdNATRAING <- juice(wdNA_rec)
          glimpse(wdNATRAING)
                    
## Model training
## Logistic regression
                    
log_spec <- logistic_reg() %>%  
set_engine(engine = "glm") %>%  
set_mode("classification") %>% 
fit(quality~., data=train_data)
                    
tidy(log_spec)
# Here we write down the results in Odds-ratio's which are better to understand.
tidy(log_spec, exponentiate=TRUE)

# only the signficant predictors on quality.
tidy(log_spec, exponentiate=TRUE) %>%
  filter(p.value < 0.05)


# Model prediction
# Now it is time to use the test data. Let us show the class predictions for the first 5 cases.
pred_class<-predict(log_spec,
                       new_data=test_data,
                       type="class")

pred_class[1:5,]

# Test data class probabilities (the predicted probabilities for the first five cases) 
pred_proba<-predict(log_spec,
                    new_data=test_data,
                    type="prob")

pred_proba[1:5,]

# Here we see the final data preparation for model evaluation

quality_results<-test_data %>%
  select(quality) %>%
  bind_cols(pred_class, pred_proba)

quality_results[1:5,]

## Model evaluation
conf_mat(quality_results, truth = quality,
         estimate = .pred_class)

# Let us now research the accuracy of the model.
accuracy(quality_results, truth=quality, 
         estimate=.pred_class)


# What can we say about the sensitivity of the model. 
sens(quality_results, truth=quality, estimate = .pred_class)


# What about the specificity of the model.
spec(quality_results, truth=quality, estimate = .pred_class)

# Now these three scores together.
custom_metrics<-metric_set(accuracy, sens, spec)

custom_metrics(quality_results,
               truth=quality,
               estimate=.pred_class)

# RESULTS SUMMARIZED
# Let us summarize the results. And it is easy to use the good and old `caret` package.

library(caret)
confusionMatrix(quality_results$.pred_class,
                quality_results$quality,
                pos="high")




                   
                   
                   
                   
                   
                   
                   