function s_idx = discretize_state(state, nbins, limits)
% Converts continuous state values into a discrete state index for Q-table
% state  = [x, x_dot, theta, theta_dot]
% nbins  = [bins_x, bins_xdot, bins_theta, bins_thetadot]
% limits = [ [x_min, x_max];
%            [x_dot_min, x_dot_max];
%            [theta_min, theta_max];
%            [theta_dot_min, theta_dot_max] ]

    bin_indices = zeros(1,4);

    for i = 1:4
        val = state(i);
        min_val = limits(i,1);
        max_val = limits(i,2);

        % Clamp value inside the range
        val = max(min_val, min(max_val, val));

        % Normalize value to [0,1]
        norm_val = (val - min_val) / (max_val - min_val);

        % Convert to a bin number between 1 and nbins(i)
        bin_indices(i) = floor(norm_val * nbins(i)) + 1;

        % Edge case: if it's exactly max, put it in last bin
        if bin_indices(i) > nbins(i)
            bin_indices(i) = nbins(i);
        end
    end

    % Convert 4 bin indices to one single row number for Q-table
    s_idx = sub2ind(nbins, bin_indices(1), bin_indices(2), ...
                             bin_indices(3), bin_indices(4));
end
