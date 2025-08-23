import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# Load data from the pickle file
try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Extract and preprocess data
try:
    data = np.asarray(data_dict['data'], dtype=object)  # Handle variable-length sequences
    labels = np.asarray(data_dict['labels'])

    # Check for empty data
    if len(data) == 0 or len(labels) == 0:
        raise ValueError("No data found in pickle file")

    print(f"Loaded {len(data)} samples with {len(set(labels))} classes")
    print("Class distribution:", Counter(labels))

except Exception as e:
    print(f"Data processing error: {e}")
    exit()

# Flatten the data while ensuring consistent dimensions
data_flattened = []
for d in data:
    try:
        # Handle both list and numpy array inputs
        landmarks = np.array(d).flatten()
        if len(landmarks) != 42:  # 21 landmarks * 2 (x,y)
            print(f"Warning: Unexpected number of features ({len(landmarks)}), expected 42")
        data_flattened.append(landmarks)
    except Exception as e:
        print(f"Skipping malformed data point: {e}")

data_flattened = np.array(data_flattened)

# Remove samples with inconsistent lengths
valid_indices = [i for i, x in enumerate(data_flattened) if len(x) == 42]
data_flattened = data_flattened[valid_indices]
labels = labels[valid_indices]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data_flattened,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels,
    random_state=42
)

# Initialize and train the model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'  # Helps with imbalanced classes
)

print("\nTraining model...")
model.fit(x_train, y_train)

# Evaluate the model
print("\nEvaluation Results:")
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f"{score * 100:.2f}% test accuracy")

print("\nClassification Report:")
print(classification_report(y_test, y_predict))

# Feature importance (for debugging)
print("\nTop 10 Most Important Features:")
importances = model.feature_importances_
top_indices = np.argsort(importances)[-10:][::-1]
print(top_indices)

# Save the trained model
try:
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    print("\nModel saved successfully as 'model.p'")
except Exception as e:
    print(f"\nError saving model: {e}")

