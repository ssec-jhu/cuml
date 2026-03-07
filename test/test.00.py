#
# test.00.py
#
# from:
#  https://docs.rapids.ai/api/cuml/stable/estimator_intro/
#

from cuml.datasets.classification import make_classification
from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score
from cuml.model_selection import train_test_split

# Generate synthetic data (binary classification task)
X, y = make_classification(
    n_classes=2,
    n_features=10,
    n_samples=1000,
    random_state=0
)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Initialize and train the model
random_forest = RandomForestClassifier(
    max_depth=10, n_estimators=25, random_state=0
).fit(X_train, y_train)

# Make predictions
predictions = random_forest.predict(X_test)

# Evaluate performance
score = accuracy_score(y_test, predictions)
print("Accuracy: ", score)
