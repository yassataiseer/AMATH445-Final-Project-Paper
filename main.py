import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
import warnings

warnings.filterwarnings("ignore")

print("1. Fetching MNIST dataset...")
# Using fetch_openml to grab standard MNIST. as_frame=False keeps it as raw numpy arrays.
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Normalize pixel values to be strictly between 0 and 1
X = X / 255.0

# Subsample the dataset to 5,000 images so the CPU can process this in minutes instead of days
X_train = X[:5000]
y_train = y[:5000]
print(f"Dataset ready. Training on {len(X_train)} samples to simulate rapid prototyping.\n")

# Define the 6 mathematical models and the hyperparameter grids we want to randomly test
models_and_params = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=100),
        "params": {"C": np.logspace(-3, 3, 100), "solver": ["lbfgs", "saga"]}
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": np.arange(1, 30), "weights": ["uniform", "distance"]}
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(),
        "params": {"max_depth": [None] + list(np.arange(5, 50, 5)), "min_samples_split": np.arange(2, 20)}
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {"n_estimators": [10, 50, 100, 200], "max_depth": [None, 10, 20, 30]}
    },
    "Linear SVM": {
        "model": LinearSVC(dual=False),
        "params": {"C": np.logspace(-3, 3, 100)}
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {"var_smoothing": np.logspace(-11, -7, 100)}
    }
}

# We will test 30 random combinations per model
n_iter_search = 30 
all_scores = {}
summary_data = []

print("2. Beginning Randomized Hyperparameter Search (This will take a few minutes)...\n")

for name, mp in models_and_params.items():
    print(f"Running 30 random configurations for {name}...")
    
    # Set up the randomized search
    random_search = RandomizedSearchCV(
        mp["model"], 
        param_distributions=mp["params"], 
        n_iter=n_iter_search, 
        cv=3, # 3-fold cross validation
        scoring='accuracy',
        random_state=42,
        n_jobs=-1 # Utilize all CPU cores for speed
    )
    
    # Train the models
    random_search.fit(X_train, y_train)
    
    # Extract the accuracy scores for every single combination tried
    scores = random_search.cv_results_['mean_test_score']
    all_scores[name] = scores
    
    # Log summary stats for the CSV table
    summary_data.append({
        "Model": name,
        "Min Accuracy": round(np.min(scores) * 100, 2),
        "Mean Accuracy": round(np.mean(scores) * 100, 2),
        "Max Accuracy (The 'Lucky Run')": round(np.max(scores) * 100, 2),
        "Variance Spread": round((np.max(scores) - np.min(scores)) * 100, 2)
    })

print("\n3. Generating Reports...")

# Export the summary data to a CSV for your Results section table
df_summary = pd.DataFrame(summary_data)
df_summary.to_csv("results_table.csv", index=False)
print("Saved data table to 'results_table.csv'")

# Convert the raw scores into a DataFrame for plotting
#df_scores = pd.DataFrame(all_scores)
df_scores = pd.DataFrame.from_dict(all_scores, orient='index').T
# Generate the Box and Whisker Plot with overlaid scatter points
plt.figure(figsize=(12, 7))
sns.boxplot(data=df_scores, palette="light:b", showfliers=False)
sns.stripplot(data=df_scores, color="darkred", alpha=0.6, jitter=True)

plt.title("Variance in MNIST Accuracy Across Random Hyperparameters", fontsize=14, pad=15)
plt.ylabel("Cross-Validation Accuracy", fontsize=12)
plt.xlabel("Machine Learning Model", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
plt.tight_layout()
plt.savefig("hyperparameter_variance_distributions.png", dpi=300)
print("Saved visualization to 'hyperparameter_variance_distributions.png'\n")
print("Process Complete. You can now use these files for your report!")


from sklearn.model_selection import validation_curve

print("4. Generating the Hyperparameter Sensitivity Curve...")
# We will use the Decision Tree for this, since it had the craziest variance in your data.
# We will test how its accuracy changes smoothly as we increase the 'max_depth' from 1 to 50.
param_range = np.arange(1, 50, 2)
train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(random_state=42), 
    X_train, 
    y_train, 
    param_name="max_depth", 
    param_range=param_range,
    cv=3, 
    scoring="accuracy", 
    n_jobs=-1
)

# Calculate the mean and standard deviation for the test scores
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.title("Hyperparameter Sensitivity: Decision Tree 'max_depth'", fontsize=14)
plt.xlabel("Maximum Tree Depth", fontsize=12)
plt.ylabel("Cross-Validation Accuracy", fontsize=12)

# Plot the accuracy curve
plt.plot(param_range, test_scores_mean, label="Test Accuracy", color="darkred", marker='o')
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="darkred")

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("sensitivity_curve.png", dpi=300)
print("Saved visualization to 'sensitivity_curve.png'")


print("5. Generating the Random Forest Pixel Importance Heatmap...")
# We train a quick standard Random Forest to see which pixels it thinks are most important
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Extract the importance of all 784 pixels and reshape them back into a 28x28 image grid
importances = rf.feature_importances_
importance_grid = importances.reshape(28, 28)

plt.figure(figsize=(8, 6))
plt.title("What the Random Forest 'Sees': Pixel Importance Heatmap", fontsize=14)
# Use a color map where dark blue = ignored pixel, bright yellow/white = highly important pixel
sns.heatmap(importance_grid, cmap="inferno", xticklabels=False, yticklabels=False)

plt.tight_layout()
plt.savefig("pixel_importance_heatmap.png", dpi=300)
print("Saved visualization to 'pixel_importance_heatmap.png'")
print("\nAll extra visualizations are complete!")


