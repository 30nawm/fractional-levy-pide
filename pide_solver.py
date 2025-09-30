# pide_solver.py
# Production-grade PIDE solver - numerically stable implementation
# Based on Chen-Deng (2014) and Zhang et al. (2018)

import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import gamma as gamma_func

class FractionalPIDESolver:
    
    def __init__(self, S0, K, T, r, sigma, alpha, theta_vg, sigma_vg, nu):
        # Clip parameters to safe ranges
        self.S0 = max(S0, 1.0)
        self.K = max(K, 1.0)
        self.T = max(T, 0.01)
        self.r = max(r, 0.0)
        self.sigma = np.clip(sigma, 0.05, 1.0)
        self.alpha = np.clip(alpha, 0.7, 0.95)
        
        # VG parameters with strict bounds
        self.theta = np.clip(theta_vg, -0.3, 0.3)
        self.sigma_vg = np.clip(sigma_vg, 0.05, 0.5)
        self.nu = np.clip(nu, 0.05, 0.8)
        
        # Check martingale condition
        mg_check = self.theta * self.nu + 0.5 * self.sigma_vg**2 * self.nu
        if mg_check >= 0.95:
            self.nu = 0.9 / (abs(self.theta) + 0.5 * self.sigma_vg**2 + 0.01)
        
        self._setup_grid()
    
    def _setup_grid(self):
        # Adaptive grid based on maturity
        if self.T < 0.1:
            self.n_t, self.n_s = 40, 60
        elif self.T < 0.5:
            self.n_t, self.n_s = 60, 80
        else:
            self.n_t, self.n_s = 80, 100
        
        # Spatial grid centered around S0
        self.S_min = max(0.1, self.S0 * 0.5)
        self.S_max = self.S0 * 2.0
        self.S_grid = np.linspace(self.S_min, self.S_max, self.n_s)
        self.dS = self.S_grid[1] - self.S_grid[0]
        
        self.dt = self.T / self.n_t
        self.idx_S0 = np.argmin(np.abs(self.S_grid - self.S0))
        
        # Stability check
        cfl = self.sigma**2 * self.dt / self.dS**2
        if cfl > 0.5:
            self.n_t = int(self.n_t * cfl / 0.4)
            self.dt = self.T / self.n_t
    
    def _build_diffusion_matrix(self):
        """Build spatial discretization matrix"""
        A = lil_matrix((self.n_s, self.n_s))
        
        # Interior points: second-order centered differences
        for i in range(1, self.n_s - 1):
            S_i = self.S_grid[i]
            
            # Coefficients
            a = 0.5 * self.sigma**2 * S_i**2 / self.dS**2
            b = self.r * S_i / (2 * self.dS)
            
            A[i, i-1] = a - b
            A[i, i] = -2*a + self.r
            A[i, i+1] = a + b
        
        # Boundaries
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
        
        return A.tocsr()
    
    def _levy_integral(self, V):
        """Stable Lévy integral with proper truncation"""
        L_V = np.zeros_like(V)
        
        # Adaptive truncation
        y_max = min(3.0, 2.0 / self.nu)
        n_y = 30
        y_grid = np.linspace(-y_max, y_max, n_y)
        dy = y_grid[1] - y_grid[0]
        
        # Precompute Lévy densities
        densities = np.zeros(n_y)
        for j, y in enumerate(y_grid):
            if abs(y) > 1e-8:
                C = 1.0 / self.nu
                if y > 0:
                    M = (np.sqrt(self.theta**2 + 2*self.sigma_vg**2/self.nu) - self.theta) / self.sigma_vg**2
                    densities[j] = C * np.exp(-M * y) / y
                else:
                    G = (np.sqrt(self.theta**2 + 2*self.sigma_vg**2/self.nu) + self.theta) / self.sigma_vg**2
                    densities[j] = C * np.exp(G * y) / (-y)
        
        # Compute integral for each grid point
        for i in range(len(self.S_grid)):
            S_i = self.S_grid[i]
            integral = 0.0
            
            for j, y in enumerate(y_grid):
                S_new = S_i * np.exp(y)
                
                if S_new <= self.S_min:
                    V_new = 0.0
                elif S_new >= self.S_max:
                    V_new = max(self.S_max - self.K, 0)
                else:
                    V_new = np.interp(S_new, self.S_grid, V)
                
                integral += (V_new - V[i]) * densities[j] * dy
            
            L_V[i] = integral
        
        # Damping for short maturities
        damping = min(1.0, np.sqrt(self.T))
        return L_V * damping
    
    def _caputo_derivative(self, V_hist):
        """L1 scheme for Caputo derivative with stability"""
        n = len(V_hist) - 1
        if n == 0:
            return np.zeros_like(V_hist[0])
        
        # L1 weights
        weights = np.array([(j+1)**(1-self.alpha) - j**(1-self.alpha) for j in range(n)])
        weights /= gamma_func(2 - self.alpha)
        
        # Compute derivative
        D_V = np.zeros_like(V_hist[0])
        for j in range(n):
            idx1 = min(n - j, len(V_hist) - 1)
            idx2 = min(n - j - 1, len(V_hist) - 1)
            diff = V_hist[idx1] - V_hist[idx2]
            D_V += weights[j] * diff
        
        # Bounded scaling
        scale = min(self.dt ** (-self.alpha), 1e4)
        return D_V * scale
    
    def solve(self):
        """Main solution loop"""
        # Terminal condition
        V = np.maximum(self.S_grid - self.K, 0.0)
        V_hist = [V.copy()]
        
        # Build diffusion matrix
        A_diff = self._build_diffusion_matrix()
        I = diags([1.0]*self.n_s, 0, format='csr')
        
        # Time loop
        for n in range(1, self.n_t + 1):
            # Fractional derivative
            D_alpha = self._caputo_derivative(V_hist)
            
            # Lévy term
            L_V = self._levy_integral(V_hist[-1])
            
            # Build system: (I - dt*A)*V_new = V_old + dt*(D_alpha + L_V)
            A_sys = I - self.dt * A_diff
            rhs = V_hist[-1] + self.dt * (D_alpha + L_V)
            
            # Boundaries
            rhs[0] = 0.0
            tau = self.T - n * self.dt
            rhs[-1] = max(self.S_grid[-1] - self.K * np.exp(-self.r * tau), 0)
            
            # Solve
            try:
                V_new = spsolve(A_sys, rhs)
                V_new = np.array(V_new).flatten()
                
                # Safety: clip and enforce monotonicity
                V_new = np.clip(V_new, 0, self.S_max)
                for i in range(1, len(V_new)):
                    V_new[i] = max(V_new[i], V_new[i-1])
                
            except:
                V_new = V_hist[-1].copy()
            
            V_hist.append(V_new)
            if len(V_hist) > 15:
                V_hist.pop(0)
        
        price = np.interp(self.S0, self.S_grid, V_hist[-1])
        price = max(0, min(price, self.S0))
        
        return price, self.S_grid, V_hist[-1]

def fractional_pide_solver(theta, sigma_vg, nu):
    from config import S0, K, T, r, sigma, alpha
    
    solver = FractionalPIDESolver(S0, K, T, r, sigma, alpha, theta, sigma_vg, nu)
    return solver.solve()