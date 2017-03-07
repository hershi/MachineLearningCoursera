data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

alphas = [0.03 0.3 0.7 0.8 0.9];
colors = ['r', 'g', 'b', 'k', 'y'];

s = length(alphas);
num_iters = 30;

% Init Theta and Run Gradient Descent 

[Xn mu sigma] = featureNormalize(X);
Xn = [ones(m,1), Xn];

figure;
for i = 1:s,
    theta = zeros(3, 1);
    [theta, J_history] = gradientDescentMulti(Xn, y, theta, alphas(i), num_iters);
    plot(1:numel(J_history), J_history, colors(i), 'LineWidth', 2);
    hold on
end

xlabel('Number of iterations');
ylabel('Cost J');
