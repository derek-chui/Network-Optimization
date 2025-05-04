%plan:
%- make 10 random points
%- sort these 10 points relative to origin (1 closest)
%- cost matrix for hungarian
%- make brute force pairing O(n!!) with points
%- all pairing functions
%- show brute force pairs on graph
%- show results in command window
%- HELPER FUNCTIONS
%- all pairing functions
%- calc tot dist for pairing set
%- generates all unique pairings (brute force) recursively
%- print pairings
%- compare results to brute force


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

    %cost matrix for hungarian
    %divide into close and far
    close = 1:(numPoints/2);
    far = (numPoints/2 + 1):numPoints;
    %build cost matrix Cij = dist(i,j) or Ri + Rj (for now, use Euclidean dist)
    costMatrix = zeros(length(close), length(far)); %#close x #far with 0
    for i = 1:length(close) %iterate close and far users
        for j = 1:length(far)
            idxA = close(i); %get user indices
            idxB = far(j);
            %dist = cost of pairing two users
            costMatrix(i, j) = norm(points(idxA,:) - points(idxB,:));
        end
    end

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

    %all pairing functions
    set1 = [1, 10; 2, 9; 3, 8; 4, 7; 5, 6];
    totDist1 = calcTotDist(set1, points);

    set2 = [1, 6; 2, 7; 3, 8; 4, 9; 5, 10];
    totDist2 = calcTotDist(set2, points);

    DNOMA = DNOMAPairing(numPoints);
    totDistDNOMA = calcTotDist(DNOMA, points);
    
    DNLUPA = DNLUPAPairing(numPoints);
    totDistDNLUPA = calcTotDist(DNLUPA, points);
    
    MUG = MUGPairing(numPoints);
    totDistMUG = calcTotDist(MUG, points);

    LCG = LCGPairing(points);
    totDistLCG = calcTotDist(LCG, points);
    DEC = DECPairing(points);
    totDistDEC = calcTotDist(DEC, points);

    hungarianPairs = hungarianPairing(costMatrix, close, far);
    totDistHungarian = calcTotDist(hungarianPairs, points);
    jvPairs = JVPairing(costMatrix, close, far);
    totDistJV = calcTotDist(jvPairs, points);

    greedyPairs = greedyPairing(points);
    totDistGreedy = calcTotDist(greedyPairs, points);
    
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
    results.BruteForce     = struct('pairs', bestPairs,        'totalDist', maxTotDist,       'complexity', 'O(n!!)');
    results.set1           = struct('pairs', set1,             'totalDist', totDist1,         'complexity', 'O(1)');
    results.set2           = struct('pairs', set2,             'totalDist', totDist2,         'complexity', 'O(1)');
    results.DNOMA          = struct('pairs', DNOMA,            'totalDist', totDistDNOMA,     'complexity', 'O(nlogn)');
    results.DNLUPA         = struct('pairs', DNLUPA,           'totalDist', totDistDNLUPA,    'complexity', 'O(nlogn)');
    results.MUG            = struct('pairs', MUG,              'totalDist', totDistMUG,       'complexity', 'O(n)');
    results.LCG            = struct('pairs', LCG,              'totalDist', totDistLCG,       'complexity', 'O(nlogn)');
    results.DEC            = struct('pairs', DEC,              'totalDist', totDistDEC,       'complexity', 'O(nlogn)');
    results.Hungarian      = struct('pairs', hungarianPairs,   'totalDist', totDistHungarian, 'complexity', 'O(n^3)');
    results.JV             = struct('pairs', jvPairs,          'totalDist', totDistJV,        'complexity', 'O(n^3)');
    results.Greedy         = struct('pairs', greedyPairs,      'totalDist', totDistGreedy,    'complexity', 'O(n^3)');
    
    printPairingResults(results);
    compareBruteForce(results);
end

%HELPER FUNCTIONS

%all pairing functions
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
    %pairing neighbours
    pairs = [];
    for i = 1:2:N-1 %1, 3, 5..
        pairs = [pairs; i, i+1]; %pair 1 with 2, 3 with 4..
    end
    %if odd, last point pairs with nearest available earlier point
    if mod(N,2) ~= 0
        pairs(end,2) = N;
    end
end

function pairs = LCGPairing(points)
    N = size(points, 1);
    [~, order] = sort(sqrt(sum(points.^2, 2))); % sort by distance to origin
    pairs = [];
    for i = 1:2:N-1
        pairs = [pairs; order(i), order(i+1)];
    end
end
function pairs = DECPairing(points)
    N = size(points, 1);
    strengths = sum(points, 2); % total strength = x + y + z
    [~, strong] = sort(strengths, 'descend');
    [~, weak] = sort(strengths, 'ascend');
    pairs = [];
    for i = 1:N/2
        pairs = [pairs; strong(i), weak(i)];
    end
end

function pairs = hungarianPairing(costMatrix, close, far)
    %minimize tot cost, far = small cost, close = large cost
    maxVal = max(costMatrix(:));
    costMatrix = maxVal - costMatrix;
    [match, ~] = matchpairs(costMatrix, 1e6); %large number to reject bad matches
    pairs = [close(match(:,1))', far(match(:,2))'];
end
function pairs = JVPairing(costMatrix, close, far)
    %matchpairs from optimization toolbox / custom LAP
    %costMatrix like hungarian
    costMatrix = -costMatrix;
    [pairIdx, ~] = matchpairs(costMatrix, -1); %-1 for max
    pairs = [];
    for i = 1:size(pairIdx,1) %pair clusters
        pairs = [pairs; close(pairIdx(i,1)), far(pairIdx(i,2))];
    end
end

function pairs = greedyPairing(points)
    N = size(points, 1);
    indices = 1:N;
    pairs = [];
    while length(indices) > 1
        maxDist = -inf;
        bestPair = [];
        %brute force check all remaining possible point pairs
        for i = 1:length(indices)
            for j = i+1:length(indices)
                idx1 = indices(i);
                idx2 = indices(j);
                dist = norm(points(idx1,:) - points(idx2,:));
                if dist > maxDist %update if better
                    maxDist = dist;
                    bestPair = [idx1, idx2];
                end
            end
        end
        %save best pair, remove from list
        pairs = [pairs; bestPair];
        indices(indices == bestPair(1)) = [];
        indices(indices == bestPair(2)) = [];
    end
end

%calc tot dist for pairing set
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

%print pairings
function printPairingResults(results)
    fields = fieldnames(results);
    for i = 1:length(fields)
        key = fields{i};
        pairStruct = results.(key);
        disp(' ');
        disp([key, ' Set ', pairStruct.complexity, ':']);
        disp(pairStruct.pairs);
        disp(['Total Distance: ', num2str(pairStruct.totalDist)]);
    end
end

%compare results to brute force
function compareBruteForce(results)
    bruteForceDist = results.BruteForce.totalDist;
    methods = fieldnames(results);

    disp(' ');
    for i = 1:length(methods)
        method = methods{i};
        if strcmp(method, 'BruteForce')
            continue;
        end
        delta = bruteForceDist - results.(method).totalDist;
        disp(['Brute Force v ', method, '? ', num2str(delta)]);
    end
    disp('The higher the numbers, the more effective the brute force algorithm is compared to other algorithms.');
end
