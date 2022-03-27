"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import pandas as pd
import numpy as np
import joblib
import os


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score,roc_curve


# Split the dataframe into test and train data
def split_data(df):
    # X = df.drop('Y', axis=1).values
    # y = df['Y'].values

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=0)
    # data = {"train": {"X": X_train, "y": y_train},
    #         "test": {"X": X_test, "y": y_test}}
    # return data
    LABEL = 'Survived'
    y_raw = df[LABEL]
    X_raw = df.drop([LABEL], axis=1)
    
     # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.3, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
             "test": {"X": X_test, "y": y_test}}
    return data


def buildpreprocessorpipeline(X_raw):
    categorical_features = X_raw.select_dtypes(include=['object']).columns
    numeric_features = X_raw.select_dtypes(include=['float','int64']).columns

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value="missing")),
                                              ('onehotencoder', OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore'))])
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ], remainder="drop")
    
    return preprocessor

# Train the model, return the model
def train_model(data, ridge_args):
    # reg_model = Ridge(**ridge_args)
    # reg_model.fit(data["train"]["X"], data["train"]["y"])
    # return reg_model

    
    lg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
    preprocessor = buildpreprocessorpipeline(data["train"]["X"])
    
    #estimator instance
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', lg)])

    model = clf.fit(data["train"]["X"],  data["train"]["y"])


# Evaluate the metrics for the model
def get_model_metrics(model, data):
    # preds = model.predict(data["test"]["X"])
    # mse = mean_squared_error(preds, data["test"]["y"])
    # metrics = {"mse": mse}
    # return metrics
    y_hat = model.predict(data["test"]["X"])
    acc = np.average(y_hat == data["test"]["Y"])

    y_scores = model.predict_proba(data["test"]["X"])
    auc = roc_auc_score(data["test"]["Y"],y_scores[:,1])
    metrics = {"acc": acc, "auc": auc }
    return metrics


def main():
    print("Running train.py")

    # # Define training parameters
    # ridge_args = {"alpha": 0.5}

    # # Load the training data as dataframe
    # data_dir = "data"
    # data_file = os.path.join(data_dir, 'diabetes.csv')
    # train_df = pd.read_csv(data_file)

    # data = split_data(train_df)

    # # Train the model
    # model = train_model(data, ridge_args)

    # # Log the metrics for the model
    # metrics = get_model_metrics(model, data)
    # for (k, v) in metrics.items():
    #     print(f"{k}: {v}")


if __name__ == '__main__':
    main()
