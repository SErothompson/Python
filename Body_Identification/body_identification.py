import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample data: player height, weight, and build type (e.g., "tall", "short", "stocky")
data = {
    'height': [180, 175, 190, 165, 185],
    'weight': [75, 70, 80, 65, 78],
    'build': ['tall', 'short', 'tall', 'short', 'stocky']
}

df = pd.DataFrame(data)

# Convert build types to numerical labels
build_labels = {'short': 0, 'tall': 1, 'stocky': 2}
df['build_label'] = df['build'].map(build_labels)

# Split data into features and labels
X = df[['height', 'weight']]
y = df['build_label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Predict build for a new player
new_player = [[178, 72]]  # Example: height 178 cm, weight 72 kg
prediction = model.predict(new_player)
build_type = list(build_labels.keys())[list(build_labels.values()).index(prediction[0])]
print(f'Predicted build: {build_type}')