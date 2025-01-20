import numpy as np
import matplotlib.pyplot as plt

def create_test_patterns(size=4):
    patterns = {
        'horizontal_line': np.zeros((size, size)),
        'vertical_line': np.zeros((size, size)),
        'diagonal': np.zeros((size, size)),
        'cross': np.zeros((size, size)),
        'empty': np.zeros((size, size)),
        'full': np.ones((size, size)),
    }
    
    # Create horizontal line pattern
    patterns['horizontal_line'][size//2,:] = 1
    
    # Create vertical line pattern
    patterns['vertical_line'][:,size//2] = 1
    
    # Create diagonal pattern
    np.fill_diagonal(patterns['diagonal'], 1)
    
    # Create cross pattern
    patterns['cross'][size//2,:] = 1
    patterns['cross'][:,size//2] = 1
    
    return patterns

def visualize_pattern(pattern):
    plt.imshow(pattern, cmap='binary')
    plt.axis('off')
    plt.show()

def generate_noisy_variants(pattern, num_variants=3, noise_level=0.1):
    variants = []
    for _ in range(num_variants):
        noisy = pattern.copy()
        noise = np.random.random(pattern.shape) < noise_level
        noisy = np.logical_xor(noisy, noise).astype(float)
        variants.append(noisy)
    return variants

def create_test_dataset():
    # Create base patterns
    patterns = create_test_patterns()
    
    # Initialize test dataset
    X = []  # Input patterns
    y = []  # Labels
    
    # Add original patterns and noisy variants
    for pattern_name, pattern in patterns.items():
        # Add original pattern
        X.append(pattern)
        y.append(pattern_name)
        
        # Add noisy variants
        noisy_variants = generate_noisy_variants(pattern)
        X.extend(noisy_variants)
        y.extend([pattern_name] * len(noisy_variants))
    
    return np.array(X), np.array(y)

# Generate dataset
X, y = create_test_dataset()

# Example of how to reshape data for neural network input
X_reshaped = X.reshape(len(X), -1)  # Flatten each 4x4 image to 16-element vector

print(X)
print(y)