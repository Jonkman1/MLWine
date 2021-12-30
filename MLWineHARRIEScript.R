# Harrie Jonkman
# MLRegression on Wine data
# 30-12-21



# install packages


# ensure pacman is installed
if(!require("pacman")) installed.packages("pacman")

# install packages from CRAN 
############################
pacman::p_load(
  
tinytex,
tidyverse,
finalfit,
tidymodels,
corrplot,
caret,
rpart, 
ranger,
vip,
randomForest)



# PROBLEM DEFINITION



# DATA LOADING, PREPROCESSING, EXPLORING

wf<-readRDS("wine.rds")

colnames(wf)

colnames(wf) <- wf %>% 
  colnames() %>% str_replace_all(pattern = " ", replacement = "_")
colnames(wf)


glimpse(wf)
summary(wf)
missing_glimpse(wf)

wf <- na.omit(wf)

missing_glimpse(wf)

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

formula <- formula(quality ~ fixed_acidity + volatile_acidity + citric_acid + 
                   residual_sugar + chlorides + free_sulfur_dioxide + 
                   total_sulfur_dioxide + density + pH + sulphates + alcohol)

lm_fit <- 
  linear_reg() %>% 
  set_mode("regression") %>% 
  set_engine("lm") %>% 
  fit(formula, data = train_data) 


print(lm_fit$fit)


summary(lm_fit$fit)

tidy(lm_fit$fit) %>% mutate_if(is.numeric, round, 3)


## 2. Decision tree

dt_fit <- 
  decision_tree() %>% 
  set_mode("regression") %>%
  set_engine("rpart") %>% 
  fit(formula, data = train_data)

print(dt_fit$fit)

library(visNetwork)
library(sparkline)
visTree(dt_fit$fit)

## 3. Random forest
rf_fit <- 
  rand_forest() %>% 
  set_mode("regression") %>%
  set_engine("randomForest") %>% 
  fit(formula, data = train_data)

print(rf_fit$fit)


# EVALUATION AND PREDICTION



## 1. Accuracy of the lm-model

lm_pred <- test_data %>% 
  bind_cols(predict(object = lm_fit, new_data = test_data))

View(lm_pred)


lm_pred <- test_data %>% 
  bind_cols(predict(object = lm_fit, new_data = test_data)) %>% 
  mutate(pred = round(.pred, 0))



lm_mse <- lm_pred %>% 
  summarise(type = "lm",
            MSE = round(mean((pred - quality)^2), 4))

head(lm_mse)


## 2. Accuracy of the Decision Tree Model

dt_pred <- test_data %>% 
  bind_cols(predict(object = dt_fit, new_data = test_data)) %>% 
  rename(pred = .pred) %>% 
  mutate(pred = round(pred, 0))

dt_mse <- dt_pred %>% 
  summarise(type = "dt",
            MSE = round(mean((pred - quality)^2), 4))


head(dt_mse)


## 3. Accuracy of the Random Forest Model

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

head(res)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Now we choosed the random forest model, we can look at the importance of the ten independent variables and compare them with each other. We see that alcohol is the most import predictor for quality followed by sulphates ad volatile_acidity. Residul-sugar, pH and fixed_acidity are the lowest important predictors for quality of wine.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ```{r}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            library(vip)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            vip(rf_fit)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ```
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Let us look at which percentage of the test sample are wrongly predicted.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ```{r, include=F}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ifelse(rf_pred$quality==rf_pred$pred, "Yes", "No")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ```
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            $105/307*100= 34,2%$ is not correctly predicted. So $65,8%$ is predicted correctly with this model. We choose the random_forest model as the best opportunity here. Let us look at it once again.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ```{r}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            head(rf_pred)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ```
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            # CONCLUSION
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            In this simple scenario, we were interested in seeing how the model performs on the testing data that were left out. The code fitted the model to the training data and apply it to the testing data. There are other ways we could have done this, but the way we do it here will be useful when we start using more complex models where we need to tune model parameters. Root Mean Square Error (RMSE) is a standard way to measure the error of a model in predicting quantitative data. RMSE is a good measure to use if we want to estimate the standard deviation of a typical observed value from our model’s prediction, R-squared is a statistical measure that represents the goodness of fit of a regression model. The ideal value for r-square is 1. The closer the value of r-square to 1, the better is the model fitted. In Machine Learning, MAE is a model evaluation metric often used with regression models. After the model is fitted and applied, we collected the performance metrics and display them and show the predictions from the testing data.  34,2% is predicted wrong, which is at the end maybe a bit disappointing after all the work. But we know what the best model is for this data-set.    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            This work has some **strengths** We found a uniform and consistent way to compare models with each and to choose the best one out of them. One of the big advantages of the random forest model (which is choosen here) is the versality and flexibility. It can be used for both regression and classification problems. But this work has also some **limitations**. Random forest is good for predictions and regression, so this could be used by the researcher for interpretation here. But for of modelling is relatively new for this researcher (instead of linear regression for example) so he found himself restricted here at the end. A limitation of random forest is also that this algorithm is fast to train, but quit slow to create predictions once they are trained: a more accurate prediction needs more trees, which results in a slower model. And a last limitation which we have to mention here is, that we used only three models and maybe other models were better for these data. 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Lesson learned is that we found a consitent workflow for analyzing data as presented here on quality of wine. It is a very good starting point for further research. The next step would be now to work on increasing the predictive power of the model and start with tuning on the hyperparameters. 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            # References
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Attalides, N. (2020). [Introduction to machine learning. Barcelona. Presentation](https://www.barcelonar.org/workshops/BarcelonaR_Introduction_to_Machine_Learning.pdf) 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Barter, R.(2020). [Tidymodels: tidy machine learning in R](http://www.rebeccabarter.com/blog/2020-03-25_machine_learning/) 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Baumer, B. Kaplan, D.T., Horton, N.J. (2017). *Modern data science with R*. CRCPress: Boca Raton.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Boehmke, B. & Greenwell, B. (2020). *Hands on machine learning with R*. [Bookdown version](https://bradleyboehmke.github.io/HOML)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Cortez, P., Cerdeira, A., Almeida, F., Matos, T. & Reis, J. (2009). Modeling wine prefernces by data mining from physicochemical properties. *Decision Support Systems, 47*, 547-533. 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Hartie, T., Tibskirani, R. & Friedmann, J. (2009). *The elements of statistical learning. Data mining, inference and prediction*. 2nd edition. Springer: New York.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Irizarry, R.A. (2020). *Introduction to data science. Data analysis and prediction algorithms with R*. CRC Press: Boca Raton.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            James, S., Witten, D., Hastie, T.. & Tibskirani, R. (2013). *An introduction to statistical learning with application in R*. 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Jonkman, H. (2019-2021). [Harrie's hoekje, his website with different blogs on machine learning in Dutch](http://www.harriejonkman.nl/HarriesHoekje/) 

Kuhn, M. 7 Johnson, K. (2013). *Applied predictive modeling*. Springer: New York.

Kuhn, M. & Johnson, K. (2019). [Feature engineering and selection: A practical approach for predictive models](www.feat.engineering)

Kuhn, M. & Silge, J. (2021). Tydy modeling with R. [Bookdown version](https://www.tmwr.org/)

Lendway, L. (2020). 2020_north-tidymodels.[Introduction on github](https://github.com/llendway/2020_north_tidymodels)

Lewis, J.E. (2020). [Coding machine learning models](https://www.youtube.com/watch?v=WlL44_is4TU)

Raoniar, R. (2021). Modeling binary logistic regression using tidymodels library in R (Part 1). [Towards data science](https://towardsdatascience.com/modelling-binary-logistic-regression-using-tidymodels-library-in-r-part-1-c1bdce0ac055)

Ruiz, E. (2019). [A gentle introduction to tidymodels](https://rviews.rstudio.com/2019/06/19/a-gentle-intro-to-tidymodels/)

Seyedian, A. (2021). [Medical cost personal datasets. Insurance forcast by using linear regression](https://www.r-bloggers.com/2021/02/using-tidymodels-to-predict-health-insurance-cost/)

Silge, J. (2020). [Get started with tidymodels and #TidyTuesdag Palmer penguins](https://juliasilge.com/blog/palmer-penguins/)

Silge, J. (2021). [Supervised machine learning case studies in R](https://supervised-ml-course)

Tidymodels. [Tidymodels website](https://www.tidymodels.org/)







