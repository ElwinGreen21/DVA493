clear; clc;

% --- Hyperparameters ---
alpha = 0.05;         % learning rate
gamma = 0.995;        % discount factor
epsilon = 1.0;       % exploration rate
epsilon_min = 0.01;
epsilon_decay = 0.999;
n_episodes = 10000;
max_steps = 3000;    % 60 seconds / 0.02

% --- Discretization setup ---
nbins = [8, 8, 32, 16];
limits = [ -2.4,  2.4;
           -3.0,  3.0;
           -0.5,  0.5;
           -5.0,  5.0 ];
n_states = prod(nbins);
n_actions = 2;
Q = zeros(n_states, n_actions);

for episode = 1:n_episodes
    % Reset environment
    state = [0 0 0 0];
    done = false;
    total_reward = 0;

    for step = 1:max_steps
        % Pick action (ε-greedy)
        s_idx = discretize_state(state, nbins, limits);
        if rand < epsilon
            action = randi([1 2]);
        else
            [~, action] = max(Q(s_idx, :));
        end

        % Convert to force and simulate
        force = 10; if action==1, force=-10; end
        next_state = SimulatePendel(force, state(1), state(2), state(3), state(4));

        % Check limits
        done = (abs(next_state(1))>2.4) || (abs(next_state(3))>12*pi/180);

        % Reward
        reward = done * (-100) + (~done)*1;

        % Update Q
        s_next = discretize_state(next_state, nbins, limits);
        Q(s_idx, action) = Q(s_idx, action) + alpha * ...
            (reward + gamma * max(Q(s_next,:)) - Q(s_idx, action));

        state = next_state;
        total_reward = total_reward + reward;

        if done, break; end
    end

    % Decay exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay);

    fprintf('Episode %d ended with reward %.1f, epsilon %.3f\n', ...
             episode, total_reward, epsilon);
end

save('Qtable.mat','Q','nbins','limits');
