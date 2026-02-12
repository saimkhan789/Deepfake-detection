import numpy as np
import matplotlib.pyplot as plt

# Load data
X = np.load("X.npy")
y = np.load("y.npy")

print("="*60)
print("DATA QUALITY ANALYSIS")
print("="*60)

print(f"\nData shapes:")
print(f"  X: {X.shape}")
print(f"  y: {y.shape}")

print(f"\nClass distribution:")
real_count = np.sum(y == 0)
fake_count = np.sum(y == 1)
print(f"  Real (0): {real_count} ({real_count/len(y)*100:.1f}%)")
print(f"  Fake (1): {fake_count} ({fake_count/len(y)*100:.1f}%)")
print(f"  Imbalance ratio: 1:{fake_count/real_count:.2f}")

print(f"\nFeature statistics:")
print(f"  Min: {X.min():.4f}")
print(f"  Max: {X.max():.4f}")
print(f"  Mean: {X.mean():.4f}")
print(f"  Std: {X.std():.4f}")

# Check for NaN or Inf
print(f"\nData quality checks:")
print(f"  Contains NaN: {np.isnan(X).any()}")
print(f"  Contains Inf: {np.isinf(X).any()}")
print(f"  Contains zeros: {(X == 0).any()}")

# Check if features differ between classes
X_real = X[y == 0]
X_fake = X[y == 1]

print(f"\nMean feature values by class:")
print(f"  Real mean: {X_real.mean():.4f}")
print(f"  Fake mean: {X_fake.mean():.4f}")
print(f"  Difference: {abs(X_real.mean() - X_fake.mean()):.4f}")

if abs(X_real.mean() - X_fake.mean()) < 0.01:
    print("\n⚠️ WARNING: Classes have very similar features!")
    print("   This might indicate a problem with feature extraction.")

print("\n" + "="*60)

# Visualization (optional - requires matplotlib)
try:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Class distribution
    axes[0, 0].bar(['Real', 'Fake'], [real_count, fake_count], color=['green', 'red'])
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].set_ylabel('Count')
    
    # Plot 2: Feature distribution (first feature, first frame)
    axes[0, 1].hist(X_real[:, 0, 0], bins=30, alpha=0.5, label='Real', color='green')
    axes[0, 1].hist(X_fake[:, 0, 0], bins=30, alpha=0.5, label='Fake', color='red')
    axes[0, 1].set_title('Feature Distribution (Sample)')
    axes[0, 1].legend()
    
    # Plot 3: Average feature across time
    axes[1, 0].plot(X_real.mean(axis=(0, 2)), label='Real', color='green')
    axes[1, 0].plot(X_fake.mean(axis=(0, 2)), label='Fake', color='red')
    axes[1, 0].set_title('Average Features Across Frames')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].legend()
    
    # Plot 4: Feature variance
    axes[1, 1].plot(X_real.std(axis=(0, 2)), label='Real', color='green')
    axes[1, 1].plot(X_fake.std(axis=(0, 2)), label='Fake', color='red')
    axes[1, 1].set_title('Feature Variance Across Frames')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=150)
    print("Saved visualization to 'data_analysis.png'")
except Exception as e:
    print(f"Could not create visualization: {e}")
