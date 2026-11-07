# Titanic Survival Prediction

This project uses logistic regression to predict Titanic passenger survival in Python and R. Each implementation runs in a separate Docker container and prints its results to the terminal.

## Prerequisites

- Git
- Docker installed and running
- A Kaggle account to download the dataset

Local Python and R installations are not required.

## 1. Clone the Repository

```bash
git clone https://github.com/xuanyubai2028/titanic-survival.git
cd titanic-survival
```

Run all commands below from the repository root.

## 2. Download the Data

1. Visit https://www.kaggle.com/competitions/titanic/data.
2. Sign in and accept the competition rules if prompted.
3. Download and extract the dataset.
4. Create the local data folder:

   ```bash
   mkdir -p src/data
   ```

5. Place `train.csv` and `test.csv` inside `src/data/`.

The resulting paths must be:

```text
src/data/train.csv
src/data/test.csv
```

The dataset is excluded from GitHub through `.gitignore`. The data folder must be created locally after cloning. `gender_submission.csv` is not required.

## 3. Run the Python Implementation

Build the image:

```bash
docker build -t titanic-python ./src/Python
```

Run the container:

```bash
docker run --rm -v "$(pwd)/src/data:/app/data:ro" titanic-python
```

## 4. Run the R Implementation

Build the image:

```bash
docker build -t titanic-r ./src/R
```

Run the container:

```bash
docker run --rm -v "$(pwd)/src/data:/app/data:ro" titanic-r
```

These commands use macOS/Linux shell syntax. Folder names are case-sensitive: use `Python` and `R` exactly as shown.

The data folder is mounted read-only at `/app/data` inside each container.

The first build downloads the required image and dependencies and may take several minutes. Rebuild the corresponding image after changing its code or dependency files.

## 5. Data Preparation and Model

Both implementations:

- Use `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, and `Fare` as features.
- Fill missing ages and fares using medians calculated from the training data.
- Encode male as `0` and female as `1`.
- Exclude `PassengerId`, `Name`, `Ticket`, `Cabin`, and `Embarked` from model features.
- Fit a logistic regression model.
- Report training accuracy and print predictions for all test passengers.

Python standardizes features and uses scikit-learn logistic regression with its default regularization. R uses an unregularized logistic regression through `glm()`. The R script also converts empty text fields to missing values before reporting missing-value counts.

## 6. Expected Terminal Output

With the original Kaggle files, each implementation prints:

- Training data dimensions: 891 rows and 12 columns.
- Test data dimensions: 418 rows and 11 columns.
- Missing-value counts and data preparation steps.
- Model training completion.
- Training accuracy of approximately 79.46% in the tested runs.
- All 418 test predictions, containing `PassengerId` and `Survived`.

`Survived = 0` predicts death; `Survived = 1` predicts survival.

Training accuracy measures performance on the data used to fit the model; it is not an estimate of performance on unseen passengers.

The test dataset does not contain `Survived` labels. Following the professor's clarification, test accuracy is not calculated; predictions are printed instead.

Results are printed to the terminal. No prediction file is created.

## 7. Project Structure

```text
titanic-survival/
├── README.md
├── .gitignore
└── src/
    ├── data/                 # Created locally; CSV files are ignored by Git
    ├── Python/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── main.py
    └── R/
        ├── Dockerfile
        ├── install_packages.R
        └── main.R
```

Python dependencies are version-pinned in `requirements.txt`. The R implementation uses the built-in `stats` and `utils` packages.