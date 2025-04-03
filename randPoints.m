%plan:
%- input points: 0 args = default, otherwise must have 3, numPoints even
%- make random points
%- make graph with points

function [points] = randPoints(numPoints,xRange, yRange)

    %input points: 0 args = default, otherwise must have 3, numPoints even
    if nargin == 0
        numPoints = 10;
        xRange = 10;
        yRange = 10;
    elseif nargin ~= 3
        error('Need 3 arguments if using input (numPoints, xRange, yRange)');
    elseif mod(numPoints, 2) ~= 0
        error('Number of points must be even');
    end
    
    %make random points
    x = rand(1, numPoints) * xRange;
    y = rand(1, numPoints) * yRange;
    points = [x; y]';
    disp('Random Points');
    disp(points);

    %make graph with points
    figure;
    scatter(x, y, 100, 'filled');
    xlabel('X');
    ylabel('Y');
    title(num2str(numPoints), ' Random Points');
    xlim([0 xRange]);
    ylim([0 yRange]);
    grid on;
    hold on;
end

%Next: Implement brute force
