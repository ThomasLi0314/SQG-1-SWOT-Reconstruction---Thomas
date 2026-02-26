% Benchmark: lsqnonlin vs fminunc for a Matrix-Output Cost Function

% 1. Setup the Synthetic Problem
rng(42); % Set random seed for reproducibility
n = 100;  % Matrix dimension
% The "true" parameters we want the solvers to find
x_true = [15; -5; 2.0];

% Generate random constant coefficient matrices
A = randn(n, n);
B = randn(n, n);
C = randn(n, n);

% Our mathematical model that produces a matrix based on parameters x
predict_matrix = @(x) x(1)*A * B + (x(2)^2)* B ^2 + exp(x(3))*C;

% Generate the target "data" matrix we want to fit (with slight noise)
M_target = predict_matrix(x_true) + 0.1*randn(n,n);

% 2. Define the Objective Functions
% ---------------------------------------------------------
% LSQNONLIN: Returns the RAW residual matrix.
fun_lsq = @(x) predict_matrix(x) - M_target;

% FMINUNC: Returns a SCALAR (sum of squared residuals).
% We use (:) to flatten the matrix, then sum the squares.
fun_fmin = @(x) sum((predict_matrix(x) - M_target).^2, 'all');
% ---------------------------------------------------------

% 3. Set Initial Guess and Options
x0 = [14.99; -4.99; 1.99]; % Poor initial guess to test robustness

% Turn off display output so we only see our final benchmark printout
options_lsq = optimoptions('lsqnonlin', 'Display', 'none');
options_fmin = optimoptions('fminunc', 'Display', 'none', 'Algorithm', 'quasi-newton');

% 4. Run and Time lsqnonlin
tic;
[x_lsq, resnorm_lsq, ~, exitflag_lsq, output_lsq] = lsqnonlin(fun_lsq, x0, [], [], options_lsq);
time_lsq = toc;

disp('Lsqnonline Finished');

% 5. Run and Time fminunc
tic;
[x_fmin, fval_fmin, exitflag_fmin, output_fmin] = fminunc(fun_fmin, x0, options_fmin);
time_fmin = toc;

% 6. Display Benchmark Results
fprintf('\n================ BENCHMARK RESULTS ================\n');
fprintf('True Parameters:      [%.4f, %.4f, %.4f]\n', x_true);
fprintf('---------------------------------------------------\n');

fprintf('LSQNONLIN:\n');
fprintf('  Found Parameters:   [%.4f, %.4f, %.4f]\n', x_lsq);
fprintf('  Sum of Squares:     %.4f\n', resnorm_lsq);
fprintf('  Function Evals:     %d\n', output_lsq.funcCount);
fprintf('  Time Elapsed:       %.4f seconds\n\n', time_lsq);

fprintf('FMINUNC:\n');
fprintf('  Found Parameters:   [%.4f, %.4f, %.4f]\n', x_fmin);
fprintf('  Sum of Squares:     %.4f\n', fval_fmin);
fprintf('  Function Evals:     %d\n', output_fmin.funcCount);
fprintf('  Time Elapsed:       %.4f seconds\n', time_fmin);
fprintf('===================================================\n');