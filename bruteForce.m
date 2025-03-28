%plan:
%- 10 for numPoints, xRange, yRange if no input
%- points must be even
%- make random points
%- make graph with points
%- make all possible pairings (brute force) with helper function
%- write the helper function that generates all unique pairings (brute force) recursively
%- look through each set made (brute force) and find max total distance
%- display pairs + max total distance
%- show the best pairings on graph

function bestPairs = bruteForce(numPoints, xRange, yRange)
    if nargin < 1
        numPoints = 10; %10 for numPoints, xRange, yRange if no input
        xRange = 10;
        yRange = 10;
    elseif mod(numPoints, 2) ~= 0 %points must be even
        error('numPoints must be even');
    end
    
    %make random points
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
    xlim([0 10]);
    ylim([0 10]);
    grid on;
    hold on;
    
    %make all possible pairings (brute force) with helper function
    pointIndices = 1:numPoints;
    pairings = makePairings(pointIndices);
    maxTotDist = -inf;
    bestPairs = [];

    %look through each set made (brute force) and find max total distance
    for i = 1:length(pairings)
        pairingSet = pairings{i}; 
        %pairings has [Set 1], [Set 2], etc. which we got from makeParings
        %pairing set is [Set i] = [1 3; 2 5; 4 6], for example
        totDist = 0;
        %tot dist for this pairing set
        for j = 1:size(pairingSet, 1) %[1 3; 2 5; 4 6] = 3 loops
            idx1 = pairingSet(j, 1); %[1 3] => 1
            idx2 = pairingSet(j, 2); %[1 3] => 3
            p1 = points(idx1, :); %1 => p1 = [2.5, 4.0]
            p2 = points(idx2, :); %3 => p2 = [5.7, 7.8]
            dist = norm(p1 - p2); %sqrt((x1-x2)^2+(y1-y^2))
            totDist = totDist + dist;
        end
        %update best pairing
        if totDist > maxTotDist
            maxTotDist = totDist;
            bestPairs = pairingSet;
        end
    end

    %display pairs + max total distance
    disp('Best Pairs:');
    disp(bestPairs);
    disp(['Max Total Distance: ', num2str(maxTotDist)]);

    %show the best pairings on graph (lines)
    for j = 1:size(bestPairs, 1)
        idx1 = bestPairs(j, 1);
        idx2 = bestPairs(j, 2);
        plot([points(idx1,1), points(idx2,1)], [points(idx1,2), points(idx2,2)], 'r-', 'LineWidth', 2);
    end
end

%write the helper function that generates all unique pairings (brute force) recursively
function pairings = makePairings(indices) %(n-1)!! ways to form pairs = O(n!!) complexity
    if isempty(indices) %base case
        pairings = {[]};
        return;
    end

    pairings = {};
    first = indices(1);
    for i = 2:length(indices) %loop over all possible choices for pairing with first
        second = indices(i);
        remaining = indices([2:i-1, i+1:end]); %exclude 1 2, get remaining
        subPairings = makePairings(remaining); %all possible pairings with remaining (recursive)
        for j = 1:length(subPairings) %each pair from recursion
            pairing = [[first, second]; subPairings{j}]; %get [1 3; 2 5; 4 6] like example above
            pairings{end+1} = pairing; %save this set to list of all possible pairings [Set 1], [Set 2], etc.
        end
    end
end

%Next: Greedy approach for 10+ points?
