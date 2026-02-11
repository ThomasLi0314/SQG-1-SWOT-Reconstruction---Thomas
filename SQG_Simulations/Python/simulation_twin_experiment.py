import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# ==========================================
# Phase 1: Forward Step (Truth Generation)
# ==========================================

class SQG_Twin_Experiment:
    def __init__(self, N=128, L=2*np.pi, Bu=1.0, Ro=0.1, z_min=-1.0, nz=32):
        self.N = N
        self.L = L
        self.Bu = Bu
        self.Ro = Ro  # epsilon
        self.nz = nz
        self.z_min = z_min
        
        # Grid Setup
        self.x = np.linspace(0, L, N, endpoint=False)
        self.y = np.linspace(0, L, N, endpoint=False)
        self.z = np.linspace(z_min, 0, nz)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Spectral Grid
        self.dk = 2 * np.pi / L
        ks = np.fft.fftfreq(N, d=1/N) * self.dk # standard ordering
        self.kx, self.ky = np.meshgrid(ks, ks, indexing='ij')
        self.K2 = self.kx**2 + self.ky**2
        self.K = np.sqrt(self.K2)
        
        # Remove mean (k=0) to avoid division by zero
        self.K[0,0] = 1.0 # arbitrary, will mask later
        
    def generate_random_phi0_surf(self, k_peak=4, slope=-3):
        """
        Generate a random 2D surface streamfunction with a specified power spectrum.
        """
        np.random.seed(42)
        phase = np.random.uniform(0, 2*np.pi, (self.N, self.N))
        amplitude = (self.K / k_peak) ** (slope) * np.exp(-(self.K/k_peak)**2)
        amplitude[0,0] = 0
        
        phi0_hat = amplitude * np.exp(1j * phase)
        phi0_surf = np.fft.ifft2(phi0_hat).real
        
        # Normalize Energy or Amplitude for reasonable numbers
        phi0_surf = phi0_surf / np.std(phi0_surf)
        return phi0_surf

    def derive_phi0_3d(self, phi0_surf):
        """
        Derive the 3D structure of Phi^0 assuming SQG balance:
        Laplacian(Phi) + 1/Bu * Phi_zz = 0
        Solution: Phi_hat(z) = Phi_hat(0) * exp( sqrt(Bu) * K * z )
        """
        phi0_surf_hat = np.fft.fft2(phi0_surf)
        
        phi0_3d = np.zeros((self.N, self.N, self.nz))
        
        # Vertical decay constant mu = sqrt(Bu) * K
        mu = np.sqrt(self.Bu) * self.K
        
        for k in range(self.nz):
            # z is negative, increasing to 0
            decay = np.exp(mu * self.z[k])
            phi0_layer_hat = phi0_surf_hat * decay
            phi0_3d[:,:,k] = np.fft.ifft2(phi0_layer_hat).real
            
        return phi0_3d

    def run_forward(self):
        print("Generating Truth Dataset...")
        
        # 1. Driver
        self.phi0_surf = self.generate_random_phi0_surf()
        self.phi0_3d = self.derive_phi0_3d(self.phi0_surf)
        
        print("Phi^0 generated.")
        
    
    def _diff_spectral(self, field, axis='x'):
        """Compute spectral derivative."""
        field_hat = np.fft.fft2(field, axes=(0,1))
        
        # Handle broadcasting for 3D fields
        if field.ndim == 3:
            kx = self.kx[:, :, np.newaxis]
            ky = self.ky[:, :, np.newaxis]
        else:
            kx = self.kx
            ky = self.ky
            
        if axis == 'x':
            res_hat = 1j * kx * field_hat
        elif axis == 'y':
            res_hat = 1j * ky * field_hat
        return np.fft.ifft2(res_hat, axes=(0,1)).real

    def solve_poisson_z(self, rhs_3d):
        """
        Solve L(psi) = rhs, where L = -K^2 + 1/Bu * d^2/dz^2
        Boundary conditions: psi=0 at z=z_min and z=0 (Dirichlet).
        Solved column-by-column in spectral space (k, l).
        """
        # Transform RHS to spectral
        rhs_hat = np.fft.fft2(rhs_3d, axes=(0,1))
        
        psi_hat = np.zeros_like(rhs_hat, dtype=complex)
        
        # Finite difference matrices for Z
        dz = self.z[1] - self.z[0]
        dz2 = dz**2
        N_z = self.nz
        N_int = N_z - 2
        
        # We solve A * x = b for each (k,l)
        # 1/Bu * (u_{k+1} - 2u_k + u_{k-1})/dz^2 - K^2 u_k = rhs_k
        # u_{k+1} - 2u_k + u_{k-1} - Bu*dz^2*K^2 u_k = Bu*dz^2 * rhs_k
        # u_{k+1} + (-2 - Bu*dz^2*K^2) u_k + u_{k-1} = ...
        
        # Vectorized Thomas Algorithm (Tridiagonal Matrix Algorithm)
        # solve A x = d for many systems
        # A is tridiagonal with diagonals a (lower), b (main), c (upper)
        # All systems have same size Nz-2 (interior)
        # But diagonals depend on K.
        
        # Flatten K2 to shape (M,) where M = N*N
        K2_flat = self.K2.flatten()
        rhs_flat = rhs_hat[:,:,1:-1].reshape(-1, N_int) # Shape (M, N_int)
        
        M = K2_flat.shape[0]
        
        # Diagonals
        # Main diag b: -2 - Bu * dz^2 * K^2
        # Upper c: 1
        # Lower a: 1
        
        main_diag = -2.0 - self.Bu * dz2 * K2_flat # Shape (M,)
        # Broadcast to (M, N_int) - actually constant in Z direction?
        # Yes, standard SQG operator has constant coefs in Z.
        
        # b array shape (M, N_int)
        b = np.tile(main_diag[:, None], (1, N_int))
        a = np.ones((M, N_int-1), dtype=complex)
        c = np.ones((M, N_int-1), dtype=complex)
        d = self.Bu * dz2 * rhs_flat
        
        # Thomas Algorithm
        # 1. Forward Elimination
        # c'[i] = c[i] / b[i]
        # d'[i] = (d[i] - a[i]*d'[i-1]) / (b[i] - a[i]*c'[i-1]) ...
        
        # Standard TDMA:
        # a: lower, b: main, c: upper.
        # modified coefficients c_prime, d_prime
        
        n = N_int
        c_prime = np.zeros((M, n-1), dtype=complex)
        d_prime = np.zeros((M, n), dtype=complex)
        
        # First step
        # c_prime[:,0] = c[:,0] / b[:,0]
        # d_prime[:,0] = d[:,0] / b[:,0]
        
        # Wait, simple loop over Z is fast (N_z=32). Loop over M is slow.
        # So we vectorize over M.
        
        # We need a mutable copy of b?
        # Actually standard TDMA modifies coefficients.
        
        # Vectorized implementation:
        c_curr = np.zeros((M, n-1), dtype=complex)
        d_curr = np.zeros((M, n), dtype=complex)
        
        # Step 0
        b_0 = b[:, 0]
        c_curr[:, 0] = c[:, 0] / b_0
        d_curr[:, 0] = d[:, 0] / b_0
        
        # Forward loop
        for i in range(1, n-1):
            temp = b[:, i] - a[:, i-1] * c_curr[:, i-1]
            c_curr[:, i] = c[:, i] / temp
            d_curr[:, i] = (d[:, i] - a[:, i-1] * d_curr[:, i-1]) / temp
            
        # Last step for d (no c)
        i = n - 1
        temp = b[:, i] - a[:, i-1] * c_curr[:, i-1]
        d_curr[:, i] = (d[:, i] - a[:, i-1] * d_curr[:, i-1]) / temp
        
        # Back Substitution
        x = np.zeros((M, n), dtype=complex)
        x[:, n-1] = d_curr[:, n-1]
        
        for i in range(n-2, -1, -1):
            x[:, i] = d_curr[:, i] - c_curr[:, i] * x[:, i+1]
            
        # Reshape to (N, N, N_int)
        psi_hat[:,:,1:-1] = x.reshape(self.N, self.N, N_int)
                
        return np.fft.ifft2(psi_hat, axes=(0,1)).real

    def calculate_higher_order(self):
        """
        Calculate F1, G1, Phi1 based on Phi0.
        """
        print("Calculating Higher Order Potentials...")
        
        # 1. Compute Derivatives of Phi0
        # We need Phi0_x, Phi0_y, Phi0_z, etc.
        # Use spectral for x,y. Analytic for z (since we derived it that way).
        
        # Recalculate Phi0_z analytically to be consistent with derivation
        # Phi_hat(z) = Phi_hat(0) * exp(mu * z)
        # d/dz -> mu * Phi_hat(z)
        phi0_surf_hat = np.fft.fft2(self.phi0_surf)
        mu = np.sqrt(self.Bu) * self.K
        
        phi0_z = np.zeros_like(self.phi0_3d)
        phi0_zz = np.zeros_like(self.phi0_3d)
        
        for k in range(self.nz):
            decay = np.exp(mu * self.z[k])
            
            # Phi0_z
            f_z_hat = phi0_surf_hat * decay * mu
            phi0_z[:,:,k] = np.fft.ifft2(f_z_hat).real
            
            # Phi0_zz
            f_zz_hat = phi0_surf_hat * decay * (mu**2)
            phi0_zz[:,:,k] = np.fft.ifft2(f_zz_hat).real
            
        phi0_x = self._diff_spectral(self.phi0_3d, 'x')
        phi0_y = self._diff_spectral(self.phi0_3d, 'y')
        
        # Need mixed derivatives for Jacobians
        phi0_zx = self._diff_spectral(phi0_z, 'x')
        phi0_zy = self._diff_spectral(phi0_z, 'y')
        
        # 2. Compute RHS terms
        # L(F1) = 2/Bu * J(Phi0_z, Phi0_x)
        # J(A, B) = A_x B_y - A_y B_x
        # J(Phi0_z, Phi0_x) = (Phi0_z)_x (Phi0_x)_y - (Phi0_z)_y (Phi0_x)_x
        #                   = Phi0_zx * Phi0_xy - Phi0_zy * Phi0_xx
        
        phi0_xx = self._diff_spectral(phi0_x, 'x')
        phi0_xy = self._diff_spectral(phi0_x, 'y')
        
        jac_F = phi0_zx * phi0_xy - phi0_zy * phi0_xx
        rhs_F = (2.0 / self.Bu) * jac_F
        
        # L(G1) = 2/Bu * J(Phi0_z, Phi0_y)
        # J(Phi0_z, Phi0_y) = Phi0_zx * Phi0_yy - Phi0_zy * Phi0_yx
        phi0_yy = self._diff_spectral(phi0_y, 'y')
        phi0_yx = phi0_xy # assuming smooth
        
        jac_G = phi0_zx * phi0_yy - phi0_zy * phi0_yx
        rhs_G = (2.0 / self.Bu) * jac_G
        
        # 3. Solve for F1, G1
        self.F1 = self.solve_poisson_z(rhs_F)
        self.G1 = self.solve_poisson_z(rhs_G)
        
        # 4. Calculate Phi1
        # Using analytic hint: Phi1_int = 1/(2Bu) * (Phi0_z)^2
        # And assuming Phi1_sur = 0 for Truth
        self.Phi1 = (1.0 / (2 * self.Bu)) * (phi0_z**2)
        
        print("F1, G1, Phi1 calculated.")

    def run_forward(self):
        print("Generating Truth Dataset...")
        
        # 1. Driver
        self.phi0_surf = self.generate_random_phi0_surf()
        self.phi0_3d = self.derive_phi0_3d(self.phi0_surf)
        
        print("Phi^0 generated.")
        
        # 2. Higher Orders
        self.calculate_higher_order()
        
        # 3. SSH
        # SSH = Phi0 + eps * Phi1 at z=0
        self.ssh = self.phi0_3d[:,:,-1] + self.Ro * self.Phi1[:,:,-1]
        
        return self.ssh

    def run_inversion(self, ssh_data):
        print("Running Inversion Step...")
        
        # 1. Invert for Phi0_recon
        # Assumption: SSH ~ Phi0 (leading order)
        self.phi0_recon = self.derive_phi0_3d(ssh_data)
        
        # 2. Invert for Higher Orders
        # Refactoring logic inline for clarity/safety:
        
        # --- Recalculate Phi0 gradients (Recon) ---
        phi0_surf_hat = np.fft.fft2(ssh_data)
        mu = np.sqrt(self.Bu) * self.K
        
        phi0_z = np.zeros_like(self.phi0_recon)
        for k in range(self.nz):
            decay = np.exp(mu * self.z[k])
            f_z_hat = phi0_surf_hat * decay * mu
            phi0_z[:,:,k] = np.fft.ifft2(f_z_hat).real
            
        phi0_x = self._diff_spectral(self.phi0_recon, 'x')
        phi0_y = self._diff_spectral(self.phi0_recon, 'y')
        phi0_zx = self._diff_spectral(phi0_z, 'x')
        phi0_zy = self._diff_spectral(phi0_z, 'y')
        
        # --- RHS for F1, G1 ---
        phi0_xx = self._diff_spectral(phi0_x, 'x')
        phi0_xy = self._diff_spectral(phi0_x, 'y')
        jac_F = phi0_zx * phi0_xy - phi0_zy * phi0_xx
        rhs_F = (2.0 / self.Bu) * jac_F

        phi0_yy = self._diff_spectral(phi0_y, 'y')
        jac_G = phi0_zx * phi0_yy - phi0_zy * phi0_xy # phi0_yx = phi0_xy
        rhs_G = (2.0 / self.Bu) * jac_G
        
        # --- Solve ---
        self.F1_recon = self.solve_poisson_z(rhs_F)
        self.G1_recon = self.solve_poisson_z(rhs_G)
        
        print("Inversion Complete.")
    
    def compute_w(self, F1, G1):
        """
        Calculate vertical velocity w = eps * (F1_x + G1_y)
        """
        F1_x = self._diff_spectral(F1, 'x')
        G1_y = self._diff_spectral(G1, 'y')
        return self.Ro * (F1_x + G1_y)

    def run_validation(self):
        print("Validating...")
        
        # 1. Truth w
        # We need to compute w from the Truth F1, G1
        # (Stored in self.F1, self.G1 from forward run)
        self.w_truth = self.compute_w(self.F1, self.G1)
        
        # 2. Recon w
        self.w_recon = self.compute_w(self.F1_recon, self.G1_recon)
        
        # 3. Compare at a specific depth (e.g. z=-0.5 or mid-depth)
        z_idx = self.nz // 2
        
        w_t = self.w_truth[:,:,z_idx]
        w_r = self.w_recon[:,:,z_idx]
        
        # Metrics
        corr = np.corrcoef(w_t.flatten(), w_r.flatten())[0,1]
        rmse = np.sqrt(np.mean((w_t - w_r)**2))
        t_std = np.std(w_t)
        
        print(f"Validation at z index {z_idx} ({self.z[z_idx]:.2f}):")
        print(f"  Correlation: {corr:.4f}")
        print(f"  RMSE: {rmse:.4e}")
        print(f"  Truth Std: {t_std:.4e}")
        print(f"  Rel Error: {rmse/t_std:.4f}")
        
        return w_t, w_r

if __name__ == "__main__":
    sim = SQG_Twin_Experiment()
    
    # Run
    ssh = sim.run_forward()
    sim.run_inversion(ssh)
    w_t, w_r = sim.run_validation()
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(131)
    plt.pcolormesh(sim.x, sim.y, w_t, shading='auto', cmap='RdBu_r')
    plt.title("Truth w (z=mid)")
    plt.colorbar()
    
    plt.subplot(132)
    plt.pcolormesh(sim.x, sim.y, w_r, shading='auto', cmap='RdBu_r')
    plt.title("Recon w (z=mid)")
    plt.colorbar()
    
    plt.subplot(133)
    plt.pcolormesh(sim.x, sim.y, w_r - w_t, shading='auto', cmap='RdBu_r')
    plt.title("Difference")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("twin_experiment_results.png")
    print("Results saved to twin_experiment_results.png")
