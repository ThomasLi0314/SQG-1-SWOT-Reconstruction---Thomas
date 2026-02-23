% This function calculated the vorticity term at the surface

function zeta_s_hat = vorticity_term(phi0_s_hat, mu, kx, ky, K2, Bu)
    
    % First term % \nabla \Phi_z \cdot \nabla_{zz}

    phi0_s_x = irfft2(phi0_s_hat * (1i) .* kx);
    phi0_s_y = irfft2(phi0_s_hat * (1i) .* ky);
    phi0_s_zzx = irfft2(phi0_s_hat .* mu .* mu .* (1i) .* kx);
    phi0_s_zzy = irfft2(phi0_s_hat .* mu .* mu .* (1i) .* ky);

    I_1_temp = phi0_s_x .* phi0_s_zzx + phi0_s_y .* phi0_s_zzy;
    
    I_1 = rfft2(I_1_temp);

    % Second term  $\nabla^2 \Phi \cdot \Phi_{zz}
    phi0_s_zz = irfft2(phi0_s_hat .* mu .* mu);
    phi0_s_lap = irfft2(phi0_s_hat .* (-1) .* K2);

    I_2_temp = phi0_s_lap .* phi0_s_zz;
    I_2 = rfft2(I_2_temp);

    % Third term $2 \|\nabla \Phi_z\|^2$
    phi0_s_zx = irfft2(phi0_s_hat .* mu .* (1i) .* kx);
    phi0_s_zy = irfft2(phi0_s_hat .* mu .* (1i) .* ky);

    I_3_temp = 2 * (phi0_s_zx .^2 + phi0_s_zy .^2);
    I_3 = rfft2(I_3_temp);

    % Fourth term $ 2 \Phi_ \nabla^2 \Phi_z
    phi0_s_z = irfft2(phi0_s_hat .* mu);
    phi0_s_lap_z = irfft2(phi0_s_hat .* (-1) .* K2 .* mu);
    
    I_4_temp = 2 * phi0_s_z .* phi0_s_lap_z;
    I_4 = rfft2(I_4_temp);

    % Fiftth term % K2 / \mu \Phi_z \Phi_zz
    I_5_temp = phi0_s_z .* phi0_s_zz;
    I_5 = rfft2(I_5_temp) * K2 ./ mu;

    % Sixth term $kiy \mu \Phi_y \Phi_z$
    I_6_temp = phi0_s_y .* phi0_s_z;
    I_6 = rfft2(I_6_temp) * (1i) .* ky .* mu;

    % Seventh term $kix \mu \Phi_x \Phi_z$
    I_7_temp = phi0_s_x .* phi0_s_z;
    I_7 = rfft2(I_7_temp) * (1i) .* kx .* mu;

    %% Sum up to get the vorticity term 
    zeta_s_hat = (I_1 + I_2 + I_3 + I_4 + I_5 + I_6 + I_7) / Bu;
end
