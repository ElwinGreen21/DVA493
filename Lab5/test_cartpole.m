clear; clc; load('Qtable.mat');

dt = 0.02;          % time step
max_steps = 3000;   % 60 seconds
state = [0 0 0 0];  % start upright
done = false;
t = 0;
reward_total = 0;

for step = 1:max_steps
    % Convert to discrete index
    s_idx = discretize_state(state, nbins, limits);

    % Pick best action (no exploration)
    [~, action] = max(Q(s_idx, :));

    % Map to force
    force = 10; if action == 1, force = -10; end

    % Simulate
    next_state = SimulatePendel(force, state(1), state(2), state(3), state(4));

    % Check limits
    done = (abs(next_state(1)) > 2.4) || (abs(next_state(3)) > 12*pi/180);
    if done
        fprintf('Pole fell at %.2f seconds!\n', step*dt);
        break;
    end

    % Update
    state = next_state;
    reward_total = reward_total + 1;
    t = t + dt;
end

if ~done
    fprintf('✅ Balanced successfully for 60 seconds!\n');
else
    fprintf('❌ Fell after %.2f seconds.\n', t);
end
