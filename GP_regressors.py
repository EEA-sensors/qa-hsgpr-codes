import numpy as np

"""
Different classes are created for each regression method.
Here we consider the gaussian process regression using the exponential kernel and
Hilbert Space approximations of the Kernel.
"""
 
class Gaussian_kernel_GPR():
    """
    Class for a GP model
    """
    def __init__(self, data, sigma2, alpha=1, scale=1):
        """
        Initialize your GP class.
        Parameters
        ----------
        data : Tuple of regression data input and observation e.g. (x, y).
        sigma2 : Float of likelihood variance.
        length_scale: Float for kernel lengthscale.
        variance: Float for kernel variance.
        """
        
        self.data = data
        self.sigma2 = sigma2
        self.alpha = alpha
        self.scale = scale

    def gaussian_kernel(self, X1, X2):
        """ 
        returns the NxM kernel matrix between the two sets of input X1 and X2 
        
        arguments:
        X1    -- NxD matrix
        X2    -- MxD matrix
        alpha -- scalar 
        scale -- scalar
        
        returns NxM matrix    
        """
        alpha = self.alpha
        scale = self.scale

        d2 = np.sum((X1[:,None,:]-X2[None,:,:])**2, axis=-1)
        K = alpha*np.exp(-0.5*d2/scale**2)

        return K

    def posterior(self, Xp):
        """ 
        returns the posterior distribution of f evaluated at each of the points in Xp conditioned on (X, y)
        using the squared exponential kernel.
        """
        kernel = self.gaussian_kernel
        x_train, y_train = self.data
        sigma2 = self.sigma2
        
        Ksf = kernel(Xp,x_train)
        Kff = kernel(x_train,x_train)
        Kss = kernel(Xp,Xp)

        jitter = 1e-6
        
        mu = Ksf @ np.linalg.inv(Kff + (jitter + sigma2)*np.identity(len(x_train))) @ y_train
        Sigma = Kss - Ksf@np.linalg.inv(Kff + sigma2*np.identity(len(x_train)))@Ksf.T
        
        return mu, Sigma
        
    def GK_spectral(self, t):

        ## Spectral density of the Gaussian Kernel
        alpha = self.alpha
        scale = self.scale

        s = alpha * np.sqrt(2*np.pi*scale**2) * np.exp(-0.5*(scale*t)**2)
        return s 

class LP_approx_GPR():
    
    """
    Class for a GP model approximating the kernel as Lazaro-Gradilla paper
    (2010) http://jmlr.org/papers/v11/lazaro-gredilla10a.html
    """
    def __init__(self, data, sigma2, M, alpha=1, scale=1):
        """
        Initialize your GP class.
        Parameters
        ----------
        data : Tuple of regression data input and observation e.g. (x, y).
        sigma2 : Float of likelihood variance.
        alpha: Amplitude of the kernel.
        scale: Float for kernel lengthscale.
        M: Approximation degree.
        samples: random samples from the spectral density of the Gaussian Kernel.
        """
        self.data = data
        self.sigma2 = sigma2
        self.alpha = alpha
        self.scale = scale
        self.M = M
        self.samples = np.random.normal(loc=0, scale=(1/(2*np.pi*scale)), size=M)
    
    def kernel_approx(self, X1, X2):
        """ 
        returns the NxM kernel matrix between the two sets of input X1 and X2 
        
        arguments:
        X1    -- NxD matrix
        X2    -- MxD matrix
    
        returns NxM matrix    
        """
    
        M = self.M
        samples = self.samples
        sigma02 = self.alpha

        d = np.sum((X1[:,None,:]-X2[None,:,:]), axis=-1)

        k = 0
        for sr in samples:
            Knew =  np.cos(2*np.pi*sr*d)
            k = Knew + k
        
        return k * sigma02/M

    def phi_vector(self, x, samples):
        """
        returns the vector of the trigonometric functions evaluated at x.
        """
        phi = None

        for sr in samples:
            phi = np.append(phi, np.array([np.cos(2*np.pi*sr*x), np.sin(2*np.pi*sr*x)]))
        phi = phi[1:]
        
        return phi[:,None]

    def Phi_matrix(self, X, samples):
        """
        returns the matrix of the trigonometric functions evaluated at each of the points in X.
        """
        phi_vector = self.phi_vector

        return (np.reshape(np.apply_along_axis(phi_vector, 1, X, samples), (len(X) , 2*len(samples))).T).astype('float64')

    def posterior(self, Xp):
        """ 
        returns the posterior distribution of f evaluated at each of the points in Xp conditioned on (X, y)
        using the approximated squared exponential kernel and the reduced notation.
        """
                
        np.random.seed(123)

        x_train, y_train = self.data
        sigma2 = self.sigma2
        alpha = self.alpha
        M = self.M

        samples = self.samples
        jitter = 1.0E-10




        phisf = self.Phi_matrix(Xp, samples)
        Phif = self.Phi_matrix(x_train, samples)
        
        A = (alpha * Phif @ Phif.T + (jitter + M * sigma2) * np.identity(2 * M)) / alpha

        A_1 = np.linalg.inv(A)
        mu = phisf.T @ A_1 @ Phif @ y_train
        
        Var = sigma2 * phisf.T @ A_1 @ phisf
        
        return mu, Var

class HS_approx_GPR():
    """
    Class for Hilbert Space approximation of the GPR based on the paper
    from Solin and Särkkä (2020) https://link.springer.com/article/10.1007/s11222-019-09886-w
    """
    def __init__(self, data, sigma2, M, L, alpha=1, scale=1):
        """
        Initialize the Hilbert Space approximation GP class.
        Parameters
        ----------
        data : Tuple of regression data input and observation e.g. (x, y).
        sigma2 : Float of likelihood variance.
        alpha: Amplitude of the kernel.
        scale: Float for kernel lengthscale.
        M: Approximation degree.
        """
        self.data = data
        self.sigma2 = sigma2
        self.alpha = alpha
        self.scale = scale
        self.M = M
        self.L = L

    def phi_j(self, lam_j, x):
        """
        returns the eigenfunction j evaluated ax the point x.
        """
        L = self.L
        return np.sin((lam_j)*(x+L))/(np.sqrt(L))
    
    def spectral_density(self, t):
        """
        Evaluates the spectral density of the exponential kernel ar the point t.
        """
        ## Spectral density of the Gaussian Kernel
        alpha = self.alpha
        scale = self.scale

        s = alpha * np.sqrt(2*np.pi*scale**2) * np.exp(-0.5*(scale*t)**2)
        return s
    
    def kernel_approx(self, X1, X2):
        """ 
        returns the NxM kernel matrix between the two sets of input X1 and X2 
        
        arguments:
        X1    -- NxD matrix
        X2    -- MxD matrix
    
        returns NxM matrix    
        """
    
        M = self.M
        alpha = self.alpha
        scale = self.scale
        L = self.L
        spectral_density = self.spectral_density
        phi_j = self.phi_j


        k = 0
        for j in range(M):
            lam_j = np.pi*(j+1)/(2*L)
            Phi_x1 = phi_j(X1,lam_j,L)
            Phi_x2 = phi_j(X2,lam_j,L)

            s_j = spectral_density(lam_j, alpha, scale)
            Knew = s_j * np.kron(Phi_x1, Phi_x2.T)
            k = Knew + k
        
        return k

    def phi_vector(self, x):
        """
        Returns the vector of x evaluated at each of the eigenfunctions from 1 to M.
        """
        M = self.M
        L = self.L
        phi_j = self.phi_j

        phi = None
        for j in range(M):
            lam_j = np.pi*(j+1)/(2*L)
            phi = np.append(phi, phi_j(lam_j, x))
        phi = phi[1:]
        
        return phi[:,None].astype('float64')

    def Phi_matrix(self, X):
        """
        returns the data matrix of the input points X evaluated at each of the eigenfunctions from 1 to M.
        """
        M = self.M
        phi_vector = self.phi_vector
        
        return np.reshape(np.apply_along_axis(phi_vector, 1, X), (len(X) , M))
    
    
    def Lambda(self):
        """
        returns the diagonal matrix of the eigenvalues.
        """
        M = self.M
        L = self.L
        spectral_density = self.spectral_density

        lams = []
        for j in range(M):
            lam_j = np.pi*(j+1)/(2*L)
            lams.append(lam_j)
        lams = np.array(lams)

        Lambda = np.diag(spectral_density(lams))
        return Lambda
    

    def posterior(self, Xp):
        """ 
        returns the posterior distribution of f evaluated at each of the points in Xp conditioned on (X, y)
        using the approximated squared exponential kernel using the Hilbert Space approximation.
        """
                
        np.random.seed(123)

        x_train, y_train = self.data
        sigma2 = self.sigma2
        M = self.M
        Phi_matrix = self.Phi_matrix

        jitter = 1.0E-10


        phisf = Phi_matrix(Xp).T
        Phif = Phi_matrix(x_train)


        Lambda = self.Lambda()
        X = Phif@np.sqrt(Lambda)
        X_s = np.sqrt(Lambda)@phisf
        
        y_train = y_train[:,None]       
        
        B = X.T @ X + (jitter +  sigma2)*np.identity(M)
        B_1 = np.linalg.inv(B)
        
        mu = X_s.T @ B_1 @ X.T @ y_train      
        Var = sigma2 * X_s.T @ B_1 @ X_s
        
        return mu, Var