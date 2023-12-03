import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the first 2000 rows of the dataset (assuming it's in an xlsx file)
data = pd.read_excel('C:/Users/USER/Downloads/predict/Inventory System.xlsx', nrows=2000)

# Features and target variable
X = data[['Product ID', 'Sales', 'Buying Price', 'Quantity']]

# Adjusted threshold: Restock if Quantity is less than 0 OR Sales are greater than 800
data['Restock'] = ((data['Quantity'] < 0) | (data['Sales'] > 800)).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['Restock'], test_size=0.2, random_state=42)

# Create a decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Display the 'Product Name', 'Prediction', and 'Actual Restock' columns
result_df = pd.DataFrame({'Product Name': data.loc[X_test.index, 'Product Name'], 'Prediction': predictions, 'Actual Restock': y_test})
print(result_df)

# Display products that need restocking
restocking_df = result_df[result_df['Prediction'] == 1]
print("\nProducts that Need Restocking:")
print(restocking_df)

# Display products that don't need restocking
no_restock_df = result_df[result_df['Prediction'] == 0]
print("\nProducts that Don't Need Restocking:")
print(no_restock_df)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'\nAccuracy: {accuracy:.2f}')

# Display classification report with zero_division='warn'
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division='warn'))

# Display decision tree rules with more descriptive class names
tree_rules = export_text(model, feature_names=X.columns.tolist(), class_names=["Do Not Restock", "Need Restocking"])
print("\nDecision Tree Rules:")
print(tree_rules)

# Visualize the decision tree and save it to a PNG file
plt.figure(figsize=(20, 15))
plot_tree(model, filled=True, feature_names=X.columns, class_names=["Do Not Restock", "Need Restocking"], rounded=True, precision=2, impurity=False, node_ids=True, fontsize=8)

# Annotate tree nodes with product names
for index, product_name in zip(X_test.index, data.loc[X_test.index, 'Product Name']):
    plt.text(model.apply(X_test.loc[[index]])[0], index, f'\n{product_name}', verticalalignment="center", horizontalalignment="left", fontsize=6)

plt.savefig('decision_tree.png')

# Show the plot (optional)
plt.show()
