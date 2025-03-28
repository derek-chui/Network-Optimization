function [points] = randPoints(numPoints,xRange, yRange)
    %random number of points
    x = rand(1, numPoints) * xRange;
    y = rand(1, numPoints) * yRange;
    points = [x; y]';
    disp('Random Points / Devices:');
    disp(points);

    %make graph with points
    figure;
    scatter(x, y, 100, 'filled');
    xlabel('X');
    ylabel('Y');
    title('Random Points / Devices');
    xlim([0 xRange]);
    ylim([0 yRange]);
    grid on;
    hold on;
end