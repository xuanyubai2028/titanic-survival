# Q23: Load and preview the Titanic datasets
library(stats)
library(utils)

cat("Q23: Loading Titanic datasets...\n")

train <- read.csv("/app/data/train.csv", stringsAsFactors = FALSE)
test <- read.csv("/app/data/test.csv", stringsAsFactors = FALSE)

cat("Training data:", nrow(train), "rows,", ncol(train), "columns\n")
cat("Test data:", nrow(test), "rows,", ncol(test), "columns\n")

cat("\nTraining preview:\n")
print(head(train))

# Q24 / Q14: Inspect missing values and survival counts
# Treat empty text fields as missing values
train[train == ""] <- NA
test[test == ""] <- NA
cat("\nQ24 / Q14: Converted empty text fields to NA.\n")

cat("\nQ24 / Q14: Missing values in training data:\n")
print(colSums(is.na(train)))

cat("\nQ24 / Q14: Missing values in test data:\n")
print(colSums(is.na(test)))

cat("\nQ24 / Q14: Survival counts (0 = died, 1 = survived):\n")
print(table(train$Survived))

# Q24 / Q14: Prepare model features
features <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare")

train_model <- train[, c("Survived", features)]
test_model <- test[, features]

# Fill missing values using training medians
for (column in c("Age", "Fare")) {
  median_value <- median(train_model[[column]], na.rm = TRUE)

  train_model[[column]][is.na(train_model[[column]])] <- median_value
  test_model[[column]][is.na(test_model[[column]])] <- median_value

  cat("\nQ24 / Q14: Filled missing", column,
      "with training median:", median_value, "\n")
}

# Encode sex
train_model$Sex <- ifelse(train_model$Sex == "female", 1, 0)
test_model$Sex <- ifelse(test_model$Sex == "female", 1, 0)

cat("\nQ24 / Q14: Encoded Sex as male=0, female=1.\n")
cat("Q24 / Q14: Selected features:", paste(features, collapse = ", "), "\n")
cat("Q24 / Q14: Excluded PassengerId, Name, Ticket, Cabin, and Embarked.\n")
cat("Q24 / Q14: Remaining missing training features:",
    sum(is.na(train_model[, features])), "\n")
cat("Q24 / Q14: Remaining missing test features:",
    sum(is.na(test_model)), "\n")

# Q24 / Q15: Train a logistic regression model
cat("\nQ24 / Q15: Training logistic regression.\n")

model <- glm(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare,
  data = train_model,
  family = binomial(link = "logit")
)

cat("Q24 / Q15: Model training completed.\n")

# Q24 / Q16: Measure training accuracy
train_probabilities <- predict(
  model,
  newdata = train_model,
  type = "response"
)

train_predictions <- as.integer(train_probabilities >= 0.5)

train_accuracy <- mean(train_predictions == train_model$Survived)

cat(sprintf(
  "\nQ24 / Q16: Training accuracy: %.2f%%\n",
  train_accuracy * 100
))

# Q24 / Q17: Predict survival on the test dataset
test_probabilities <- predict(
  model,
  newdata = test_model,
  type = "response"
)

test_predictions <- as.integer(test_probabilities >= 0.5)

prediction_results <- data.frame(
  PassengerId = test$PassengerId,
  Survived = test_predictions
)

cat("\nQ24 / Q17: Generated predictions for",
    nrow(prediction_results), "passengers.\n")
print(prediction_results, row.names = FALSE)

# Q24 / Q18: Follow the professor's clarification
cat("\nQ24 / Q18: Test accuracy is unavailable because test.csv has no Survived labels.\n")