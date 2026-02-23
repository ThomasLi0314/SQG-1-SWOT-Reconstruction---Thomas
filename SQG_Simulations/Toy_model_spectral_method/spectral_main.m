%% This is the main file
clear;
clc;
close all;

% Some important parameters can be changed here
% grid size
Nx = 128;
Ny = 128;

% Domain size
Lx = 2 * pi;
Ly = 2 * pi;

Ro = 0.01; %Rossby number

initialize;

%% Define the initial phi0_file
phi0_s = cos(X) + cos(Y);

% Compute the fourier transform 
phi0_s_hat = fft2(phi0_s);

%% Forward Part
% This part derives the true fourier SSH 

cyclogeo_term_true = cyclogeo_term(phi0_s_hat, kx, ky);
vorticity_term_true = vorticity_term(phi0_s_hat, mu, kx, ky, K2, Bu);

% True pressure field 
p1_s_hat_true = (f * cyclogeo_term_true + vorticity_term_true) .* Kn2;

eta_s_hat_true = f * phi0_s_hat + p1_s_hat_true * epsilon; 

fprintf('True SSH data generated\n');

%% Inversion part

% Initial guess very close to the true value
max_phi0_s = max(phi0_s(:));
phi0_s_guess = phi0_s + 0.001 * max_phi0_s * randn(Nx, Ny);
phi0_s_hat_guess = fft2(phi0_s_guess);

% Optimization options
num_iteration = 20;

% Parallel computing
if isempty(gcp('nocreate'))
    parpool;
end

% No parallel
% options = optimoptions('lsqnonlin', 'Display', 'iter', 'Algorithm', 'trust-region-reflective', 'MaxIterations', num_iteration);

% Parallel
options = optimoptions('lsqnonlin', 'Display', 'iter', 'Algorithm', 'trust-region-reflective', 'MaxIterations', num_iteration, 'UseParallel', true);

% Compute the cost function
cost_func = @(phi0_s_hat_guess) cost_function(phi0_s_hat_guess, kx, ky, mu, Bu, K2, eta_s_hat_true);

% Run the optimization
tic;
try
    [phi0_s_hat_opt, resnorm] = lsqnonlin(cost_func, phi0_s_hat_guess, [], [], options);
    disp('Optimization Complete.');
catch ME
    disp('Optimization filer or interrupted.')
    disp(ME.message);
end
toc;

%% Forward Second round to obtain the velocity field.

% Calculate the surface u
[u_surface_opt, v_surface_opt] = calculate_surface_u(phi0_s_hat_opt, mu, kx, ky, K2, Kn2, epsilon, Bu);

% Calculate the true surface u
[u_surface_true, v_surface_true] = calculate_surface_u(phi0_s_hat, mu, kx, ky, K2, Kn2, epsilon, Bu);

%% Plotting the difference
figure('Position', [100, 100, 1200, 400]);

% Plot Optimized Surface U
subplot(1, 3, 1);
imagesc(x, y, u_surface_opt');
set(gca, 'YDir', 'normal');
colorbar;
title('Optimized u_{surface}');
xlabel('x'); ylabel('y');
axis equal tight;

% Plot True Surface U
subplot(1, 3, 2);
imagesc(x, y, u_surface_true');
set(gca, 'YDir', 'normal');
colorbar;
title('True u_{surface}');
xlabel('x'); ylabel('y');
axis equal tight;

% Plot Difference
subplot(1, 3, 3);
u_diff = u_surface_opt - u_surface_true;
imagesc(x, y, u_diff');
set(gca, 'YDir', 'normal');
colorbar;
title('Difference (Opt - True)');
xlabel('x'); ylabel('y');
axis equal tight;

colormap jet;
sgtitle('Surface Zonal Velocity Comparison');
saveas(gcf, 'u_surface_comparison.png');
