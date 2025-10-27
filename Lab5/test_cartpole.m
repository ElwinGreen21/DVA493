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

% --- Förbättrad animation av pendeln med hjul som vilar på rälsen ---
figure('Color','w','Name','Inverted Pendulum Visualization');
cart_width = 0.4;
cart_height = 0.2;
pole_length = 0.5;
wheel_radius = 0.05;

% Förbered figuren
axis equal
axis([-3 3 -1 1]);
hold on;
xlabel('Cart Position (m)');
ylabel('Height (m)');
title('Inverted Pendulum Control Visualization');
grid on;

% Rita golv (ljusgrön bakgrund)
railY = -cart_height/2 - wheel_radius; % Rälsen direkt under hjulen
fill([-3 3 3 -3], [-1 -1 railY-0.02 railY-0.02], [0.9 1 0.9], 'EdgeColor','none');

% Räls (lite ovanför golvet)
rail = line([-3 3], [railY railY], 'Color', [0.3 0.3 0.3], 'LineWidth', 2);

% Skapa objekt som uppdateras
cart = rectangle('Position',[X(1)-cart_width/2, -cart_height/2, cart_width, cart_height], ...
                 'FaceColor',[0.2 0.6 1], 'EdgeColor','k', 'Curvature',0.2);

% Hjulen ska vila på rälsen
theta_circ = linspace(0, 2*pi, 20);
wheelL = fill(X(1)-cart_width/3 + wheel_radius*cos(theta_circ), ...
              railY + wheel_radius*sin(theta_circ), [0 0 0]);
wheelR = fill(X(1)+cart_width/3 + wheel_radius*cos(theta_circ), ...
              railY + wheel_radius*sin(theta_circ), [0 0 0]);

% Pendeln
pole = line([X(1), X(1) + pole_length*sin(Theta(1))], ...
            [cart_height/2, cart_height/2 + pole_length*cos(Theta(1))], ...
            'LineWidth',4,'Color',[0.1 0.1 0.1]);

% Tidstext
timeText = text(-2.8, 0.85, 'Time: 0.00 s', 'FontSize',12, 'FontWeight','bold', 'Color',[0 0 0.6]);

% --- Animation loop ---
for i = 1:length(X)
    % Uppdatera vagnens position
    set(cart, 'Position', [X(i)-cart_width/2, -cart_height/2, cart_width, cart_height]);
    
    % Uppdatera hjulens position (nu på rälsen)
    set(wheelL, 'XData', X(i)-cart_width/3 + wheel_radius*cos(theta_circ), ...
                'YData', railY + wheel_radius*sin(theta_circ));
    set(wheelR, 'XData', X(i)+cart_width/3 + wheel_radius*cos(theta_circ), ...
                'YData', railY + wheel_radius*sin(theta_circ));
    
    % Uppdatera pendelns topposition
    px = X(i) + pole_length * sin(Theta(i));
    py = cart_height/2 + pole_length * cos(Theta(i));
    set(pole, 'XData', [X(i) px], 'YData', [cart_height/2 py]);
    
    % Uppdatera tidstext
    set(timeText, 'String', sprintf('Time: %.2f s', i*dt));
    
    drawnow limitrate;
    pause(0.005);
end

% Slutmeddelande
if ~done
    msg = sprintf('✅ Balanced for %.2f s!', length(X)*dt);
    text(-1,0.9,msg,'FontSize',14,'FontWeight','bold','Color',[0 0.5 0]);
else
    msg = sprintf('❌ Fell after %.2f s', length(X)*dt);
    text(-1,0.9,msg,'FontSize',14,'FontWeight','bold','Color',[0.8 0 0]);
end
