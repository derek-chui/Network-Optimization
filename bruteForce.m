%plan:
%- make 10 random points
%- sort these 10 points relative to origin (1 closest)
%- make brute force pairing O(n!!) with points
%- make comparison pairing O(1) with points
%- show brute force pairs on graph
%- show results in command window
%- HELPER FUNCTIONS
%- tot dist for this pairing set
%- generates all unique pairings (brute force) recursively

function bruteForce()
    numPoints = 10;
    xRange = 10;
    yRange = 10;
    
    %make random points
    x = rand(1, numPoints) * xRange;
    y = rand(1, numPoints) * yRange;
    points = [x; y]';

    %sort these 10 points relative to origin (1 closest)
    dist = sqrt(sum(points.^2, 2));
    [~, sortedIndices] = sort(dist);
    points = points(sortedIndices, :);

    %make brute force pairing O(n!!) with points
    pointIndices = 1:numPoints;
    pairings = makePairings(pointIndices);
    maxTotDist = -inf;
    bestPairs = [];
    %get total distances
    for i = 1:length(pairings)
        pairingSet = pairings{i};
        %pairings has [Set 1], [Set 2], etc. which we got from makeParings
        %pairing set is [Set i] = [1 3; 2 5; 4 6], for example
        totDist = calcTotDist(pairingSet, points);
        %update best pairing
        if totDist > maxTotDist
            maxTotDist = totDist;
            bestPairs = pairingSet;
        end
    end
    
    %make comparison pairing O(1) with points
    set1 = [1, 10; 2, 9; 3, 8; 4, 7; 5, 6];
    set2 = [1, 6; 2, 7; 3, 8; 4, 9; 5, 10];
    totDist1 = calcTotDist(set1, points);
    totDist2 = calcTotDist(set2, points);
    
    %show brute force pairs on graph
    figure;
    scatter(points(:,1), points(:,2), 100, 'filled');
    hold on;
    for j = 1:size(bestPairs, 1)
        idx1 = bestPairs(j, 1);
        idx2 = bestPairs(j, 2);
        plot([points(idx1,1), points(idx2,1)], [points(idx1,2), points(idx2,2)], 'r-', 'LineWidth', 2);
    end
    xlabel('X'); ylabel('Y');
    title('Brute Force Pairs');
    grid on;
    
    %show results in command window
    disp('Brute Force Set O(n!!):');
    disp(bestPairs);
    disp(['Max Total Distance: ', num2str(maxTotDist)]);
    
    disp(' ');
    disp('Set 1 O(1)');
    disp(set1);
    disp(['Total Distance: ', num2str(totDist1)]);
    
    disp(' ');
    disp('Set 2 O(1)');
    disp(set2);
    disp(['Total Distance: ', num2str(totDist2)]);
    
    disp(' ');
    disp(['How much better is Brute Force Compared to 1st Set? ', num2str(maxTotDist - totDist1)]);
    disp(['How much better is Brute Force Compared to 2nd Set? ', num2str(maxTotDist - totDist2)]);
    disp(['The higher the numbers, the more effective the brute force algorithm is compared to predicted cases.'])
end

%HELPER FUNCTIONS

%tot dist for this pairing set
function totDist = calcTotDist(pairingSet, points)
    totDist = 0;
    for j = 1:size(pairingSet, 1) %[1 3; 2 5; 4 6] = 3 loops
        idx1 = pairingSet(j, 1); %[1 3] => 1
        idx2 = pairingSet(j, 2); %[1 3] => 3
        p1 = points(idx1, :); %1 => p1 = [2.5, 4.0]
        p2 = points(idx2, :); %3 => p2 = [5.7, 7.8]
        dist = norm(p1 - p2); %sqrt((x1-x2)^2+(y1-y^2))
        totDist = totDist + dist;
    end
end

%generates all unique pairings (brute force) recursively
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
