% This code is for the spectral method inversion. The code will also consist of two part. A forward part and a backward part using the optimization. However, everything is done in the spectral space and every data is 2D. 

%% Remarks

% Everything _hat is the fourier transform. 

% Everything with _s is the surface value.

%% Define the Domain


Bu = 1; %Burger number
f = 1; %Coriolis parameter
% N = 1; % Brunt Vaisala frequency
epsilon = Ro

% grid spacing
dx = Lx / Nx;
dy = Ly / Ny;

x = (0 : Nx - 1) * dx;
y = (0 : Ny - 1) * dy;

[X, Y] = ndgrid(x, y);

% Spectral grid, spectral space should havae same grid point as the primal space. 
dk = 2 * pi / Nx;
dl = 2 * pi / Ny;

k_zonal = [0 : Nx/2 -1, -Nx/2 : -1] * dk;
l_meridional = [0 : Ny/2 -1, -Ny/2 : -1] * dl;

[kx, ky] = ndgrid(k_zonal, l_meridional);

% Wave number
K2 = sqrt(kx.^2 + ky.^2);
K2(1, 1) = 1e-13; % Avoid division by zero
Kn2 = 1 ./ K2;
Kn2(1, 1) = 1e16; % Avoid division by zero

% \mu appears in the \Phi^0 analytic solution.
mu = sqrt(Bu) * K2;
