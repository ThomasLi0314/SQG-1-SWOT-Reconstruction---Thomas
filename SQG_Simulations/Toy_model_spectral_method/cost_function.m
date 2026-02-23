function cost = cost_function(phi0_s_hat_guess, kx, ky, mu, Bu, K2, eta_s_hat_true)

    % this function compute the cost function for the optimization

    % cyclogeo term
    cyclogeo_term_guess = cyclogeo_term(phi0_s_hat_guess, kx, ky);
    vorticity_term_guess = vorticity_term(phi0_s_hat_guess, mu, kx, ky, K2, Bu);

    % p1 guess field
    p1_s_hat_guess = (f * cyclogeo_term_guess + vorticity_term_guess) .* Kn2;

    % SSH guess field
    eta_s_hat_guess = f * phi0_s_hat_guess + p1_s_hat_guess * epsilon; 

    % cost function 
    cost = abs(eta_s_hat_guess - eta_s_hat_true);
    
end