---
title: "SA2_DATA_MINING"
author: "QUINTERO & GONZALES"
date: '`r Sys.Date()`'
output: github_document
---

```{r}

library(tidyverse)
library(caret)
library(glmnet)
library(randomForest)
library(gbm)
library(pROC)
library(corrplot)

training<-read.csv("C:/Users/sbcvj/Downloads/churn-bigml-80.csv")
test<-read.csv("C:/Users/sbcvj/Downloads/churn-bigml-20.csv")

training$State <- NULL
training$Area.code<- NULL
test$State <- NULL
test$Area.code<- NULL

head(training)
```


```{r}
#Convert the churn column into binary
training$Churn <- as.numeric(factor(training$Churn, levels = c("False", "True"))) - 1
test$Churn <- as.numeric(factor(test$Churn, levels = c("False", "True"))) - 1

#Converting Categorial variable using One Hot Encoding

training$International.plan <- ifelse(training$International.plan == "Yes", 1, 0)
training$Voice.mail.plan <- ifelse(training$Voice.mail.plan == "Yes", 1, 0)


test$International.plan <- ifelse(test$International.plan == "Yes", 1, 0)
test$Voice.mail.plan <- ifelse(test$Voice.mail.plan == "Yes", 1, 0)


#Remove NA's
training<-na.omit(training)
#Remove NA's
test<-na.omit(test)



library(ggplot2)
library(scales)

ggplot(training, 
       aes(x = Churn, y= after_stat(count/sum(count)))) +
  geom_bar(fill = "cornflowerblue", 
           color = "white") + 
  labs(title="Churn Distribution - Training Set", 
       y = "Percent",
       x = "0= No, 1= Yes") +
  scale_y_continuous(labels = percent)
```


```{r}
summary(training[, c("Total.day.minutes", "Total.eve.minutes", "Total.night.minutes", "Total.intl.minutes")])
```


```{r}
call_data <- training[, c("Total.day.minutes", "Total.eve.minutes", "Total.night.minutes", "Total.intl.minutes")]
call_data_long <- reshape2::melt(call_data)

ggplot(call_data_long, aes(x = value, fill = variable)) +
  geom_histogram(bins = 30, alpha = 0.7) +
  facet_wrap(~variable, scales = "free") +
  labs(title = "Distribution of Call Minutes- Training Set", x = "Minutes", y = "Frequency")
```


```{r}
numeric_vars <- training[sapply(training, is.numeric)]
cor_matrix <- cor(numeric_vars, use = "complete.obs")

library(reshape)

melted_corrmat_pearson <- melt(cor_matrix)
head(melted_corrmat_pearson)
```

```{r}
ggplot(melted_corrmat_pearson, aes(x = X1, y = X2, fill = value)) + 
  geom_tile() + 
  labs(x = 'Variables', y = 'Variables') + 
  coord_fixed()
```

```{r}
ggplot(training, 
       aes(x = Churn, y= after_stat(count/sum(count)))) +
  geom_bar(fill = "cornflowerblue", 
           color = "white") + 
  labs(title="Churn Distribution - Training Set", 
       y = "Percent",
       x = "0= No, 1= Yes") +
  scale_y_continuous(labels = percent)
```


```{r}
# We can see that there is a significant imbalance, which can be biased if not fixed

#Balancing Churn

predictor_variables <- training[,-18] 
response_variable <- training$Churn 
levels(response_variable) <- c('0', '1') 
levels(test$Churn)<- c('0', '1') 

library(smotefamily)
library(caret)
library(nnet)

set.seed(123)
smote_data <- SMOTE(X = predictor_variables,
                    target = response_variable,
                    K = 5,              
                    dup_size = 0)        

balanced_data <- smote_data$data
balanced_data$class <- as.factor(balanced_data$class)

# Assumptions

cor_matrix <- cor(balanced_data[sapply(balanced_data, is.numeric)])
corrplot::corrplot(cor_matrix, method = "color")
```

```{r}
# We can see that Total eve / Day charge is highly correlated to total day/eve minutes aslo voice mail plan. Thus consider removing it 
library(dplyr) 

balanced_data <- balanced_data %>% 
  select(-Total.day.charge, -Total.eve.charge, -Total.night.charge, -Total.intl.charge,- Number.vmail.messages)

cor_matrix <- cor(balanced_data[sapply(balanced_data, is.numeric)])
corrplot::corrplot(cor_matrix, method = "color")
```


```{r}
# Train a logistic regression model
lr_model <- glm(class ~ ., data = balanced_data, family = "binomial")
library(car)


vif_values <- vif(lr_model)
vif_values
```

```{r}
#All VIF values are well below 5 - No strong multicollinearity

cooks_d <- cooks.distance(lr_model)
plot(cooks_d, 
     pch = 20,             # solid dots
     main = "Scatter Plot of Cook's Distance",
     xlab = "Observation Number",
     ylab = "Cook's Distance",
     col = "blue")

abline(h = 4 / length(cooks_d), col = "red", lty = 2)
```

```{r}
# No extreme outliers

summary(lr_model)
```

```{r}
# Make predictions on the test set
pred_probs <- predict(lr_model, newdata = test, type = "response")
pred_class <- ifelse(pred_probs > 0.5, 1, 0)
table(pred_class)
```

```{r}
table(Predicted = pred_class, Actual = test$Churn)
```

```{r}
pred_class <- factor(pred_class, levels = c(0, 1))
actual_class <- factor(test$Churn, levels = c(0, 1))

conf_matrix <- confusionMatrix(pred_class, actual_class, positive = "1")


conf_matrix <- confusionMatrix(pred_class, actual_class, positive = "1")

accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Sensitivity"]
f1 <- 2 * (precision * recall) / (precision + recall)

accuracy
```

```{r}
precision
```
```{r}
recall
```

```{r}
f1
```
```{r}
#Lasso Regression

library(glmnet)

x <- as.matrix(balanced_data[, -ncol(balanced_data)])
y <- balanced_data$class
set.seed(123)
cv_model <- cv.glmnet(x, y, family = "binomial", alpha = 1)  
best_lambda <- cv_model$lambda.min

lasso_regression <- glmnet(x, y, family = "binomial", alpha = 1, lambda = best_lambda)

drop_cols <- c("Churn", "Total.day.charge", "Total.eve.charge", 
               "Total.night.charge", "Total.intl.charge", "Number.vmail.messages")

test_x <- as.matrix(test[, !names(test) %in% drop_cols])
test_y <- test$Churn

pred_prob <- predict(lasso_regression, newx = test_x)
pred_class <- factor(ifelse(pred_prob >= 0.5, 1, 0), levels = c(0, 1))
test_y <- factor(test_y, levels = c("0", "1"))
pred_class <- factor(pred_class, levels = c("0", "1"))
conf_matrix_lasso <- confusionMatrix(pred_class, test_y, positive = "1")
conf_matrix_lasso
```

```{r}
accuracy <- conf_matrix_lasso$overall["Accuracy"]


precision <- conf_matrix_lasso$byClass["Precision"]

recall <- conf_matrix_lasso$byClass["Sensitivity"]

f1 <- 2 * (precision * recall) / (precision + recall)

# Print results
accuracy
```
```{r}
precision
```

```{r}
recall
```

```{r}
f1
```

```{r}
# Decision Tree Model
library(rpart)
library(rpart.plot)

dt_model <- rpart(class ~ ., data = balanced_data, method = "class")

rpart.plot(dt_model, main = "Decision Tree for Churn Prediction")

pred_dt_prob <- predict(dt_model, newdata = as.data.frame(test_x), type = "prob")[, 2]
pred_dt_class <- ifelse(pred_dt_prob >= 0.5, 1, 0)
pred_dt_class <- factor(pred_dt_class, levels = c(0, 1))

conf_matrix_dt <- confusionMatrix(pred_dt_class, test_y, positive = "1")

accuracy <- conf_matrix_dt$overall["Accuracy"]
precision <- conf_matrix_dt$byClass["Precision"]
recall <- conf_matrix_dt$byClass["Sensitivity"]
f1 <- 2 * (precision_dt * recall_dt) / (precision_dt + recall_dt)

accuracy
precision
recall
f1

```

```{r}
library(randomForest)

rf_model <- randomForest(class ~ ., data = balanced_data, ntree = 100, mtry = sqrt(ncol(balanced_data) - 1))

pred_rf_prob <- predict(rf_model, newdata = as.data.frame(test_x), type = "prob")[, 2]
pred_rf_class <- ifelse(pred_rf_prob >= 0.5, 1, 0)
pred_rf_class <- factor(pred_rf_class, levels = c(0, 1))

conf_matrix_rf <- confusionMatrix(pred_rf_class, test_y, positive = "1")

accuracy <- conf_matrix_rf$overall["Accuracy"]
precision <- conf_matrix_rf$byClass["Precision"]
recall <- conf_matrix_rf$byClass["Sensitivity"]
f1f <- 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)

accuracy
precision
recall
f1

```

```{r}
library(xgboost)

train_label <- as.numeric(as.factor(balanced_data$class)) - 1 
train_matrix <- as.matrix(balanced_data[, -which(names(balanced_data) == "class")])

xgb_model <- xgboost(data = train_matrix, label = train_label, 
                     objective = "binary:logistic", nrounds = 100)

test_matrix <- as.matrix(test[, !names(test) %in% drop_cols])
test_label <- as.numeric(as.factor(test$Churn)) - 1  # Convert to 0 and 1

pred_xgb_prob <- predict(xgb_model, newdata = test_matrix)

pred_xgb_class <- ifelse(pred_xgb_prob >= 0.5, 1, 0)
pred_xgb_class <- factor(pred_xgb_class, levels = c(0, 1))

conf_matrix_xgb <- confusionMatrix(pred_xgb_class, factor(test_label, levels = c(0, 1)), positive = "1")

accuracy<- conf_matrix_xgb$overall["Accuracy"]
precision <- conf_matrix_xgb$byClass["Precision"]
recall<- conf_matrix_xgb$byClass["Sensitivity"]
f1 <- 2 * (precision_xgb * recall_xgb) / (precision_xgb + recall_xgb)

accuracy
precision
recall
f1

```






Note that the `echo = FALSE` parameter was added to the code chunk to prevent printing of the R code that generated the plot.
