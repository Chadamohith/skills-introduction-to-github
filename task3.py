import pandas as pd
import matplotlib.pyplot as plt

# Input data
data = {
    "Outlook": ["sunny", "sunny", "overcast", "rain", "rain", "rain", "overcast", "sunny", "sunny", "rain", "sunny", "overcast", "overcast", "rain"],
    "Temperature": ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot", "mild"],
    "Humidity": ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"],
    "Windy": ["false", "true", "false", "false", "false", "true", "true", "false", "false", "false", "true", "true", "false", "true"],
    "Class": ["-", "-", "+", "+", "+", "-", "+", "-", "+", "+", "+", "+", "+", "-"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define the input query
query = {"Outlook": "sunny", "Temperature": "cool", "Humidity": "high", "Windy": "true"}

# Helper function to calculate probabilities
def calculate_probability(df, query, target_class):
    class_count = len(df[df["Class"] == target_class])
    total_count = len(df)
    class_prob = class_count / total_count

    # Calculate conditional probabilities
    feature_prob = 1
    for feature, value in query.items():
        feature_count = len(df[(df[feature] == value) & (df["Class"] == target_class)])
        feature_prob *= feature_count / class_count

    return feature_prob * class_prob

# Calculate probabilities for each class
p_positive = calculate_probability(df, query, "+")
p_negative = calculate_probability(df, query, "-")

# Normalize probabilities
p_total = p_positive + p_negative
p_positive_normalized = p_positive / p_total
p_negative_normalized = p_negative / p_total

# Output results
print("Probability of + (play tennis):", p_positive_normalized)
print("Probability of - (not play tennis):", p_negative_normalized)

# Plot results
labels = ["Play Tennis (+)", "Not Play Tennis (-)"]
probs = [p_positive_normalized, p_negative_normalized]

plt.bar(labels, probs, color=["green", "red"])
plt.ylabel("Probability")
plt.title("Prediction Probabilities")
plt.show() 