import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# Configure plot style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

# Create output directory
output_dir = "logistic_regression_outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configure matplotlib to not show plots on screen
plt.ioff()  # Disable interactive mode

class LogisticRegression:
    """
    Logistic Regression implementation with Gradient Descent
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.costs = []
        
    def sigmoid(self, z):
        """Sigmoid function"""
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def log_loss(self, y_true, y_pred):
        """Log-loss cost function"""
        # Avoid log(0) by adding small constant
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iter):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Calculate cost
            cost = self.log_loss(y, y_pred)
            self.costs.append(cost)
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check convergence
            if i > 0 and abs(self.costs[-2] - self.costs[-1]) < self.tol:
                print(f"Convergence reached at iteration {i}")
                break
    
    def predict_proba(self, X):
        """Predict probabilities"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X):
        """Make binary predictions"""
        return (self.predict_proba(X) >= 0.5).astype(int)

def create_and_dataset():
    """Create dataset for logical AND function"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # Logical AND
    return X, y

def create_or_dataset():
    """Create dataset for logical OR function"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])  # Logical OR
    return X, y

def create_circles_dataset():
    """Create dataset with concentric circles"""
    np.random.seed(42)
    n_samples = 200
    
    # Generate random points
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    
    # Create inner circle (class 0)
    n_inner = n_samples // 2
    r_inner = np.random.uniform(0, 0.3, n_inner)  # Small radius
    X_inner = np.column_stack([
        r_inner * np.cos(angles[:n_inner]),
        r_inner * np.sin(angles[:n_inner])
    ])
    y_inner = np.zeros(n_inner)
    
    # Create outer circle (class 1)
    n_outer = n_samples - n_inner
    r_outer = np.random.uniform(0.6, 1.0, n_outer)  # Large radius
    X_outer = np.column_stack([
        r_outer * np.cos(angles[n_inner:]),
        r_outer * np.sin(angles[n_inner:])
    ])
    y_outer = np.ones(n_outer)
    
    # Combine data
    X = np.vstack([X_inner, X_outer])
    y = np.concatenate([y_inner, y_outer])
    
    # Add some noise
    noise = np.random.normal(0, 0.05, X.shape)
    X += noise
    
    return X, y

def create_linear_dataset():
    """Create linearly separable dataset"""
    np.random.seed(42)
    n_samples = 200
    
    # Generate points for class 0 (bottom-left)
    n_class0 = n_samples // 2
    X_class0 = np.random.multivariate_normal(
        mean=[-1, -1], 
        cov=[[0.5, 0.2], [0.2, 0.5]], 
        size=n_class0
    )
    y_class0 = np.zeros(n_class0)
    
    # Generate points for class 1 (top-right)
    n_class1 = n_samples - n_class0
    X_class1 = np.random.multivariate_normal(
        mean=[1, 1], 
        cov=[[0.5, -0.2], [-0.2, 0.5]], 
        size=n_class1
    )
    y_class1 = np.ones(n_class1)
    
    # Combine data
    X = np.vstack([X_class0, X_class1])
    y = np.concatenate([y_class0, y_class1])
    
    return X, y

def plot_sigmoid():
    """Visualize sigmoid function"""
    z = np.linspace(-10, 10, 100)
    sigmoid_z = 1 / (1 + np.exp(-z))
    
    plt.figure(figsize=(10, 6))
    plt.plot(z, sigmoid_z, 'b-', linewidth=2, label='Sigmoid Function')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold 0.5')
    plt.axvline(x=0, color='g', linestyle='--', alpha=0.7, label='z = 0')
    plt.xlabel('z (input)')
    plt.ylabel('σ(z) (output)')
    plt.title('Sigmoid Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "sigmoid_function.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save sigmoid data
    sigmoid_data = {
        'z_values': z.tolist(),
        'sigmoid_values': sigmoid_z.tolist(),
        'description': 'Sigmoid function: σ(z) = 1/(1 + e^(-z))'
    }
    
    with open(os.path.join(output_dir, "sigmoid_data.json"), 'w') as f:
        json.dump(sigmoid_data, f, indent=2)

def plot_decision_boundary(X, y, model, title="Decision Boundary", filename=None):
    """Visualize decision boundary"""
    plt.figure(figsize=(10, 8))
    
    # Create mesh of points
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot contours
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdBu')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Plot data points
    colors = ['red' if label == 0 else 'blue' for label in y]
    scatter = plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black', s=50)
    plt.colorbar(plt.cm.ScalarMappable(cmap='RdBu'), ax=plt.gca())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'{title}')
    
    # Save plot
    if filename:
        plt.savefig(os.path.join(output_dir, f"decision_boundary_{filename}.png"), 
                   dpi=300, bbox_inches='tight')
    plt.close()

def plot_cost_evolution(costs, filename=None):
    """Visualize cost evolution"""
    plt.figure(figsize=(10, 6))
    plt.plot(costs, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Log-Loss')
    plt.title('Cost Evolution During Training')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    if filename:
        plt.savefig(os.path.join(output_dir, f"cost_evolution_{filename}.png"), 
                   dpi=300, bbox_inches='tight')
    plt.close()

def train_and_evaluate(X, y, title):
    """Train model and show results"""
    print(f"\n{'='*50}")
    print(f"TRAINING: {title}")
    print(f"{'='*50}")
    
    # Create and train model
    model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    
    # Calculate metrics
    accuracy = np.mean(y == y_pred)
    final_cost = model.costs[-1]
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Final cost: {final_cost:.4f}")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias:.4f}")
    
    # Show predictions
    print("\nPredictions:")
    for i in range(len(X)):
        print(f"X={X[i]}, y_true={y[i]}, y_pred={y_pred[i]}, prob={y_prob[i]:.4f}")
    
    # Save model results
    filename_safe = title.lower().replace(' ', '_')
    
    results = {
        'title': title,
        'accuracy': float(accuracy),
        'final_cost': float(final_cost),
        'weights': model.weights.tolist(),
        'bias': float(model.bias),
        'num_iterations': len(model.costs),
        'input_data': X.tolist(),
        'true_labels': y.tolist(),
        'predictions': y_pred.tolist(),
        'probabilities': y_prob.tolist(),
        'cost_evolution': model.costs,
        'hyperparameters': {
            'learning_rate': model.learning_rate,
            'max_iter': model.max_iter,
            'tolerance': model.tol
        }
    }
    
    # Save results to JSON
    with open(os.path.join(output_dir, f"results_{filename_safe}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save dataset to CSV
    dataset_data = np.column_stack([X, y])
    np.savetxt(os.path.join(output_dir, f"dataset_{filename_safe}.csv"), 
               dataset_data, delimiter=',', 
               header='X1,X2,y', comments='')
    
    return model

# ==========================================
# MAIN EXECUTION
# ==========================================

print("LOGISTIC REGRESSION WITH GRADIENT DESCENT")
print("=" * 60)
print(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {output_dir}")
print("=" * 60)

# Create log file
log_file = os.path.join(output_dir, "training_log.txt")
with open(log_file, 'w') as f:
    f.write("LOGISTIC REGRESSION WITH GRADIENT DESCENT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")

# 1. Visualize sigmoid function
print("\n1. VISUALIZING SIGMOID FUNCTION")
with open(log_file, 'a') as f:
    f.write("1. VISUALIZING SIGMOID FUNCTION\n")
    f.write("- Plot saved: sigmoid_function.png\n")
    f.write("- Data saved: sigmoid_data.json\n\n")

plot_sigmoid()

# 2. Toy datasets
datasets = {
    "Logical AND": create_and_dataset(),
    "Logical OR": create_or_dataset(),
    "Linearly Separable": create_linear_dataset(),
    "Concentric Circles": create_circles_dataset()
}

# 3. Train models
models = {}
performance_summary = []

for name, (X, y) in datasets.items():
    model = train_and_evaluate(X, y, name)
    models[name] = (model, X, y)
    
    # Add to summary
    accuracy = np.mean(y == model.predict(X))
    performance_summary.append({
        'dataset': name,
        'accuracy': accuracy,
        'final_cost': model.costs[-1],
        'iterations': len(model.costs)
    })

# 4. Visualize decision boundaries
print("\n" + "="*60)
print("VISUALIZING DECISION BOUNDARIES")
print("="*60)

with open(log_file, 'a') as f:
    f.write("2. VISUALIZING DECISION BOUNDARIES\n")

for name, (model, X, y) in models.items():
    filename_safe = name.lower().replace(' ', '_')
    plot_decision_boundary(X, y, model, f"Decision Boundary - {name}", filename_safe)
    
    with open(log_file, 'a') as f:
        f.write(f"- {name}: decision_boundary_{filename_safe}.png\n")

# 5. Visualize cost evolution
print("\n" + "="*60)
print("COST EVOLUTION")
print("="*60)

with open(log_file, 'a') as f:
    f.write("\n3. COST EVOLUTION\n")

plt.figure(figsize=(15, 10))
for i, (name, (model, _, _)) in enumerate(models.items(), 1):
    plt.subplot(2, 2, i)
    plt.plot(model.costs, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Log-Loss')
    plt.title(f'Cost Evolution - {name}')
    plt.grid(True, alpha=0.3)
    
    # Save individual plot
    filename_safe = name.lower().replace(' ', '_')
    plot_cost_evolution(model.costs, filename_safe)
    
    with open(log_file, 'a') as f:
        f.write(f"- {name}: cost_evolution_{filename_safe}.png\n")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cost_evolution_comparison.png"), 
           dpi=300, bbox_inches='tight')
plt.close()

# 6. Performance comparison
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)

print(f"{'Dataset':<20} {'Accuracy':<10} {'Final Cost':<12} {'Iterations':<12}")
print("-" * 55)

# Create summary for file
summary_text = []
summary_text.append("PERFORMANCE SUMMARY")
summary_text.append("="*60)
summary_text.append(f"{'Dataset':<20} {'Accuracy':<10} {'Final Cost':<12} {'Iterations':<12}")
summary_text.append("-" * 55)

for item in performance_summary:
    line = f"{item['dataset']:<20} {item['accuracy']:<10.4f} {item['final_cost']:<12.4f} {item['iterations']:<12}"
    print(line)
    summary_text.append(line)

# Save performance summary
with open(os.path.join(output_dir, "performance_summary.txt"), 'w') as f:
    f.write('\n'.join(summary_text))

# Save complete summary in JSON
complete_summary = {
    'execution_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'performance_summary': performance_summary,
    'files_generated': {
        'plots': [
            'sigmoid_function.png',
            'cost_evolution_comparison.png'
        ],
        'decision_boundaries': [f"decision_boundary_{name.lower().replace(' ', '_')}.png" 
                               for name in models.keys()],
        'cost_evolutions': [f"cost_evolution_{name.lower().replace(' ', '_')}.png" 
                           for name in models.keys()],
        'datasets': [f"dataset_{name.lower().replace(' ', '_')}.csv" 
                    for name in models.keys()],
        'results': [f"results_{name.lower().replace(' ', '_')}.json" 
                   for name in models.keys()],
        'other': [
            'sigmoid_data.json',
            'performance_summary.txt',
            'training_log.txt',
            'complete_summary.json'
        ]
    }
}

with open(os.path.join(output_dir, "complete_summary.json"), 'w') as f:
    json.dump(complete_summary, f, indent=2)

# Finalize log
with open(log_file, 'a') as f:
    f.write("\n4. GENERATED FILES\n")
    f.write("="*30 + "\n")
    total_files = (len(complete_summary['files_generated']['plots']) + 
                   len(complete_summary['files_generated']['decision_boundaries']) + 
                   len(complete_summary['files_generated']['cost_evolutions']) + 
                   len(complete_summary['files_generated']['datasets']) + 
                   len(complete_summary['files_generated']['results']) + 
                   len(complete_summary['files_generated']['other']))
    f.write(f"Total files generated: {total_files}\n")
    f.write("All files saved in: " + output_dir + "\n")

print("\nTraining completed! 🎉")
print("Observations:")
print("- AND/OR: Simple problems, converge quickly")
print("- Linear: Linearly separable, good performance")
print("- Circles: Not linearly separable, limited performance")
print(f"\n📁 All files saved in: {output_dir}/")
print(f"📊 Total files generated: {len(os.listdir(output_dir))}")
print("\n📋 Main files:")
print("- complete_summary.json: General experiment summary")
print("- training_log.txt: Detailed process log")
print("- performance_summary.txt: Performance table")
print("- *.png: Plots and visualizations")
print("- *.json: Detailed data and results")
print("- *.csv: Datasets used")ls
