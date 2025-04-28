%plan:
%- make 10 random points
%- sort these 10 points relative to origin (1 closest)
%- make brute force pairing O(n!!) with points
%- all pairing functions
%- show brute force pairs on graph
%- show results in command window
%- HELPER FUNCTIONS
%- all NOMA helper functions
%- tot dist for this pairing set
%- generates all unique pairings (brute force) recursively

function NOMA()
    numPoints = 10;
    xRange = 10;
    yRange = 10;
    zRange = 10;
    
    %make random points
    x = rand(1, numPoints) * xRange;
    y = rand(1, numPoints) * yRange;
    z = rand(1, numPoints) * zRange;
    points = [x; y; z]';

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

    %- all pairing functions
    DNOMA = DNOMAPairing(numPoints);
    totDistDNOMA = calcTotDist(DNOMA, points);
    
    DNLUPA = DNLUPAPairing(numPoints);
    totDistDNLUPA = calcTotDist(DNLUPA, points);
    
    MUG = MUGPairing(numPoints);
    totDistMUG = calcTotDist(MUG, points);

    [LCG, DEC] = LCG_DECPairing(points);
    totDistLCG = calcTotDist(LCG, points);
    totDistDEC = calcTotDist(DEC, points);
    
    %show brute force pairs on graph
    figure;
    scatter3(points(:,1), points(:,2), points(:,3), 100, 'filled');
    hold on;
    for j = 1:size(bestPairs, 1)
        idx1 = bestPairs(j, 1);
        idx2 = bestPairs(j, 2);
        plot3([points(idx1,1), points(idx2,1)], [points(idx1,2), points(idx2,2)], [points(idx1,3), points(idx2,3)], 'r-', 'LineWidth', 2);
    end
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('Brute Force Pairs');
    grid on;
    
    %show results in command window
    disp('Brute Force Set O(n!!):');
    disp(bestPairs);
    disp(['Max Total Distance: ', num2str(maxTotDist)]);

    disp('D-NOMA Set O(n log n):');
    disp(DNLUPA);
    disp(['Total Distance: ', num2str(totDistDNOMA)]);

    disp('D-NLUPA Set O(n log n):');
    disp(DNLUPA);
    disp(['Total Distance: ', num2str(totDistDNLUPA)]);

    disp(' ');
    disp('MUG Set O(n):');
    disp(MUG);
    disp(['Total Distance: ', num2str(totDistMUG)]);
    
    disp(' ');
    disp('LCG Set O(nlogn):');
    disp(LCG);
    disp(['Total Distance: ', num2str(totDistLCG)]);
    
    disp(' ');
    disp('DEC Set O(nlogn):');
    disp(DEC);
    disp(['Total Distance: ', num2str(totDistDEC)]);
    
    disp(' ');
    disp(['How much better is Brute Force Compared to D-NOMA? ', num2str(maxTotDist - totDistDNOMA)]);
    disp(['How much better is Brute Force Compared to D-NLUPA? ', num2str(maxTotDist - totDistDNLUPA)]);
    disp(['How much better is Brute Force Compared to MUG? ', num2str(maxTotDist - totDistMUG)]);
    disp(['How much better is Brute Force Compared to LCG? ', num2str(maxTotDist - totDistLCG)]);
    disp(['How much better is Brute Force Compared to DEC? ', num2str(maxTotDist - totDistDEC)]);
    disp(['The higher the numbers, the more effective the brute force algorithm is compared to predicted cases.'])
end

%HELPER FUNCTIONS

%all NOMA functions
function pairs = DNOMAPairing(N)
    %divide into 4 groups
    %FNN (First Nearest Near)
    g1 = 1:floor(N/4);
    %SNN (Second Nearest Near)
    g2 = (floor(N/4)+1):floor(N/2);
    %FFF (First Farthest Far)
    g3 = (floor(N/2)+1):floor(3*N/4);
    %SFF (Second Farthest Far)
    g4 = (floor(3*N/4)+1:N);
    
    %pair g1 with g3 (odd)
    oddPairs = [];
    len = min(length(g1), length(g3));
    for i = 1:len
        oddPairs = [oddPairs; g1(i), g3(i)];
    end
    
    %pair g2 with g4 (even)
    evenPairs = [];
    len = min(length(g2), length(g4));
    for i = 1:len
        evenPairs = [evenPairs; g2(i), g4(i)];
    end
    pairs = [oddPairs; evenPairs]; %combine
end

function pairs = DNLUPAPairing(N)
    %divide into 2 groups
    group1 = 1:floor(N/2);
    group2 = (floor(N/2)+1):N;
    
    %pair first in g1 with last in g2 etc
    pairs = [];
    for i = 1:length(group1)
        pairs = [pairs; group1(i), group2(end-i+1)];
    end
end

function pairs = MUGPairing(N)
    %appx MUG by pairing close indices
    pairs = [];
    for i = 1:2:N-1
        pairs = [pairs; i, i+1];
    end
    %if odd, last point pairs with nearest available earlier point
    if mod(N,2) ~= 0
        pairs(end,2) = N;
    end
end

function [LCG, DEC] = LCG_DECPairing(points)
    N = size(points, 1);
    % LCG, greedy sort by tot dist
    [~, order] = sort(sqrt(sum(points.^2, 2)));
    LCG = [];
    for i = 1:2:N-1
        LCG = [LCG; order(i), order(i+1)];
    end
    % DEC, alternating strongest and weakest
    [~, strong] = sort(points(:,1) + points(:,2) + points(:,3), 'descend');
    [~, weak] = sort(points(:,1) + points(:,2) + points(:,3), 'ascend');
    DEC = [];
    for i = 1:N/2
        DEC = [DEC; strong(i), weak(i)];
    end
end

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
