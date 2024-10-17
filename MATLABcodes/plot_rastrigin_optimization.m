% MATLAB script to read optimization data and plot the Rastrigin function with evaluation points
function plot_rastrigin_optimization()
    % Define the path to the data folder relative to the MATLAB code
    data_path = fullfile('..', 'data', 'optimization_data.csv');
    
    % Read the data from the CSV file
    data = readmatrix(data_path);
    x_points = data(:, 1);  % Extract x-coordinates
    y_points = data(:, 2);  % Extract y-coordinates
    values = data(:, 3);    % Extract corresponding function values

    % Define the Rastrigin function parameters
    A = 10;  % Default parameter for the Rastrigin function

    % Create a grid of points for plotting the Rastrigin function
    x = linspace(-5.12, 5.12, 400);
    y = linspace(-5.12, 5.12, 400);
    [X, Y] = meshgrid(x, y);

    % Evaluate the Rastrigin function over the grid
    Z = A * 2 + (X.^2 - A * cos(2 * pi * X)) + (Y.^2 - A * cos(2 * pi * Y));

    % Create a 3D plot of the Rastrigin function
    figure;
    surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.7); % Surface plot
    colormap('parula');  % Use the 'viridis' colormap for consistency
    hold on;

    % Overlay the evaluation points from the optimization process
    scatter3(x_points, y_points, values, 50, 'r', 'filled', 'DisplayName', 'Evaluation Points');

    % Highlight the global minimum at (0, 0)
    scatter3(0, 0, 0, 100, 'w', 'x', 'LineWidth', 2, 'DisplayName', 'Global Minimum (0, 0)');

    % Set labels and title
    title('3D Rastrigin Function with Evaluation Points');
    xlabel('x');
    ylabel('y');
    zlabel('Function Value');
    legend('Rastrigin Surface', 'Evaluation Points', 'Global Minimum (0, 0)');
    view(30, -45);  % Adjust the viewing angle for better perspective

    % Enable grid for better visualization
    grid on;

    % Save the plot as an image file
    saveas(gcf, fullfile('..', 'data', 'rastrigin_optimization_plot.png'));
    fprintf('3D plot saved as ''rastrigin_optimization_plot.png'' in the data folder.\n');
end
