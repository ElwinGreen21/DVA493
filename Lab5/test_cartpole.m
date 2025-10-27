clear; clc; load('Qtable.mat');

dt = 0.02;          
max_steps = 3000;   
state = [0 0 0 0];  
done = false;
t = 0;
reward_total = 0;

X = [];
Theta = [];

for step = 1:max_steps
    s_idx = discretize_state(state, nbins, limits);
    [~, action] = max(Q(s_idx, :));
    force = 10; if action == 1, force = -10; end
    next_state = SimulatePendel(force, state(1), state(2), state(3), state(4));

    done = (abs(next_state(1)) > 2.4) || (abs(next_state(3)) > 12*pi/180);
    if done
        fprintf('Pole fell at %.2f seconds!\n', step*dt);
        break;
    end

    % --- Logga tillstånd för visualisering ---
    X(end+1) = next_state(1);
    Theta(end+1) = next_state(3);

    state = next_state;
    reward_total = reward_total + 1;
    t = t + dt;
end

if ~done
    fprintf('✅ Balanced successfully for 60 seconds!\n');
else
    fprintf('❌ Fell after %.2f seconds.\n', t);
end

% --- Plot (statiska grafer) ---
t_axis = (0:length(X)-1)*dt;
figure;
subplot(2,1,1);
plot(t_axis, X, 'LineWidth', 1.5);
ylabel('Cart position (m)');
title('Cart position over time');
grid on;

subplot(2,1,2);
plot(t_axis, Theta * 180/pi, 'LineWidth', 1.5);
ylabel('Pole angle (deg)');
xlabel('Time (s)');
title('Pole angle over time');
grid on;

% --- Animation (valfri) ---
figure;
axis([-3 3 -1 1]);
hold on;
cart_width = 0.4;
cart_height = 0.2;
pole_length = 0.5;

for i = 1:length(X)
    clf;
    rectangle('Position',[X(i)-cart_width/2, -cart_height/2, cart_width, cart_height],'FaceColor',[0 0.5 1]);
    px = X(i) + pole_length * sin(Theta(i));
    py = cart_height/2 + pole_length * cos(Theta(i));
    line([X(i), px],[cart_height/2, py],'LineWidth',3,'Color','k');
    line([-3,3],[-cart_height/2,-cart_height/2],'Color','k');
    axis([-3 3 -1 1]);
    title(sprintf('Step %d / %d', i, length(X)));
    drawnow;
    pause(0.01);
end
