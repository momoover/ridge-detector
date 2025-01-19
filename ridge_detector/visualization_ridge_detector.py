import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec
from PIL import Image

def create_test_image(size=50):
    """Create a test image with a curved line"""
    x = np.linspace(0, size-1, size)
    y = np.linspace(0, size-1, size)
    xx, yy = np.meshgrid(x, y)
    
    # Create a curved line
    curve = np.exp(-(yy - (0.3 * xx * np.sin(xx/30) + size/2))**2 / 50)
    
    return curve

def compute_derivatives(image, sigma):
    """Compute first and second derivatives using Gaussian kernels"""
    # First derivatives
    Ix = gaussian_filter1d(image, sigma, axis=1, order=1)
    Iy = gaussian_filter1d(image, sigma, axis=0, order=1)
    
    # Second derivatives
    Ixx = gaussian_filter1d(image, sigma, axis=1, order=2)
    Iyy = gaussian_filter1d(image, sigma, axis=0, order=2)
    Ixy = gaussian_filter1d(Ix, sigma, axis=0, order=1)
    
    return Ix, Iy, Ixx, Iyy, Ixy

def compute_hessian_features(Ixx, Ixy, Iyy):
    """Compute eigenvalues and eigenvectors of the Hessian matrix"""
    shape = Ixx.shape
    lambda1 = np.zeros(shape)
    lambda2 = np.zeros(shape)
    nx = np.zeros(shape)
    ny = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            H = np.array([[Iyy[i,j], Ixy[i,j]], 
                         [Ixy[i,j], Ixx[i,j]]])
            eigenvals, eigenvecs = np.linalg.eigh(H)
            
            lambda1[i,j] = eigenvals[0]
            lambda2[i,j] = eigenvals[1]
            nx[i,j] = eigenvecs[0,0]
            ny[i,j] = eigenvecs[0,1]
    
    return lambda1, lambda2, nx, ny

def compute_ridge_points(Ix, Iy, Ixx, Ixy, Iyy, nx, ny, lambda1, lambda2, sigma, threshold=0.1):
    """Compute ridge points with subpixel accuracy"""
    ridge_points = []
    offset_vectors = []  # Store offset vectors for visualization
    
    for i in range(1, Ix.shape[0]-1):
        for j in range(1, Ix.shape[1]-1):
            # Ridge saliency
            R = sigma**2 * max(abs(lambda1[i,j]), abs(lambda2[i,j]))
            
            if R > threshold:
                # Compute offset using equation (4)
                denominator = (Ixx[i,j]*nx[i,j]**2 + 
                             2*Ixy[i,j]*nx[i,j]*ny[i,j] + 
                             Iyy[i,j]*ny[i,j]**2)
                
                if abs(denominator) > 1e-10:
                    t = -(Ix[i,j]*nx[i,j] + Iy[i,j]*ny[i,j]) / denominator
                    
                    # Subpixel position
                    x_sub = j + t*nx[i,j]
                    y_sub = i + t*ny[i,j]
                    
                    ridge_points.append((x_sub, y_sub, R))
                    offset_vectors.append((j, i, x_sub-j, y_sub-i))  # original pos + offset vector
    
    return ridge_points, offset_vectors

def visualize_detailed_steps(image, Ix, Iy, Ixx, Ixy, Iyy, lambda1, lambda2, nx, ny, ridge_points, offset_vectors, sigma):
    """Detailed visualization of all steps in the ridge detection process"""
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 4)
    
    # 1. Original Image
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image\n(σ={:.1f})'.format(sigma))
    ax1.axis('off')
    
    # 2. First Derivatives
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(np.hypot(Ix, Iy), cmap='viridis')
    ax2.set_title('Gradient Magnitude\n(First Derivatives)')
    ax2.axis('off')
    
    # 3. Second Derivatives
    ax3 = plt.subplot(gs[0, 2])
    ax3.imshow(Ixx, cmap='coolwarm')
    ax3.set_title('Second Derivative (Ixx)')
    ax3.axis('off')
    
    ax4 = plt.subplot(gs[0, 3])
    ax4.imshow(Iyy, cmap='coolwarm')
    ax4.set_title('Second Derivative (Iyy)')
    ax4.axis('off')
    
    # 4. Hessian Analysis
    ax5 = plt.subplot(gs[1, 0])
    ax5.imshow(Ixy, cmap='coolwarm')
    ax5.set_title('Cross Derivative (Ixy)')
    ax5.axis('off')
    
    # 5. Eigenvalues
    ax6 = plt.subplot(gs[1, 1])
    ax6.imshow(lambda1, cmap='coolwarm')
    ax6.set_title('First Eigenvalue (λ₁)')
    ax6.axis('off')
    
    ax7 = plt.subplot(gs[1, 2])
    ax7.imshow(lambda2, cmap='coolwarm')
    ax7.set_title('Second Eigenvalue (λ₂)')
    ax7.axis('off')
    
    # 6. Eigenvectors
    ax8 = plt.subplot(gs[1, 3])
    step = 10  # Skip pixels for clearer visualization
    y, x = np.mgrid[0:image.shape[0]:step, 0:image.shape[1]:step]
    ax8.quiver(x, y, nx[::step,::step], ny[::step,::step], 
               scale=30, color='red', width=0.003)
    ax8.imshow(image, cmap='gray', alpha=0.3)
    ax8.set_title('Eigenvector Direction')
    ax8.axis('off')
    
    # 7. Ridge Detection Steps
    ax9 = plt.subplot(gs[2, 0])
    ax9.imshow(image, cmap='gray')
    if offset_vectors:
        offsets = np.array(offset_vectors)
        ax9.quiver(offsets[:,0], offsets[:,1], 
                  offsets[:,2] * 0.005, offsets[:,3] * 0.005,
                  scale=1, color='red', width=0.003, 
                  headwidth=3, headlength=5, headaxislength=4.5)
    ax9.set_title('Subpixel Offset Vectors')
    ax9.axis('off')
    
    # 8. Ridge Points
    ax10 = plt.subplot(gs[2, 1])
    ax10.imshow(image, cmap='gray')
    if ridge_points:
        x_coords, y_coords, strengths = zip(*ridge_points)
        ax10.scatter(x_coords, y_coords, c='red', s=1)
    ax10.set_title('Detected Ridge Points')
    ax10.axis('off')
    
    # 9. Ridge Strength
    ax11 = plt.subplot(gs[2, 2])
    ax11.imshow(image, cmap='gray')
    if ridge_points:
        strengths = np.array(strengths)
        strengths_normalized = (strengths - strengths.min()) / (strengths.max() - strengths.min())
        scatter = ax11.scatter(x_coords, y_coords, c=strengths_normalized, 
                             cmap='hot', s=1)
        plt.colorbar(scatter, ax=ax11)
    ax11.set_title('Ridge Strength')
    ax11.axis('off')
    
    # 10. Final Result
    ax12 = plt.subplot(gs[2, 3])
    ax12.imshow(image, cmap='gray')
    if ridge_points:
        x_coords, y_coords, strengths = zip(*ridge_points)
        ax12.scatter(x_coords, y_coords, c='red', s=1, alpha=0.5)
    ax12.set_title('Final Ridge Detection')
    ax12.axis('off')
    
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    # Create test image
    image = create_test_image()
    # Open the image and convert to grayscale
    #image = Image.open('/Users/matthiasoverberg/Desktop/smal_test.jpg').convert('L')

    # Convert the grayscale image to a NumPy array
    #image = np.array(image)
    #image = 255 - image
    
    # Parameters
    sigma = 2.0
    threshold = 0.1
    
    # Compute derivatives
    Ix, Iy, Ixx, Iyy, Ixy = compute_derivatives(image, sigma)
    
    # Compute Hessian features
    lambda1, lambda2, nx, ny = compute_hessian_features(Ixx, Ixy, Iyy)
    
    # Detect ridge points
    ridge_points, offset_vectors = compute_ridge_points(Ix, Iy, Ixx, Ixy, Iyy, 
                                                      nx, ny, lambda1, lambda2, 
                                                      sigma, threshold)
    
    # Visualize detailed steps
    fig = visualize_detailed_steps(image, Ix, Iy, Ixx, Ixy, Iyy, lambda1, lambda2, 
                                 nx, ny, ridge_points, offset_vectors, sigma)
    plt.show()
