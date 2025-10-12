function NOMA3Q()
    rng('shuffle');
    numPoints = 12; %change number of users here
    xRange = 10; yRange = 10; zRange = 10; %change simulation dimensions here

    %make rand pts
    x = rand(1, numPoints) * xRange;
    y = rand(1, numPoints) * yRange;
    z = rand(1, numPoints) * zRange;
    weights = rand(1, numPoints);
    points = [x; y; z]';

    %sort pts from origin
    dist0 = sqrt(sum(points.^2, 2));
    [~, sortedIndices] = sort(dist0);
    points  = points(sortedIndices, :);
    weights = weights(sortedIndices);

    %brute force
    pointIndices = 1:numPoints;
    groupings = makeGroupings(pointIndices);
    maxScore = -inf; bestTriplets = [];
    for i = 1:length(groupings)
        groupingSet = groupings{i};
        sc = calcScore(groupingSet, points, weights);
        if sc > maxScore
            maxScore = sc;
            bestTriplets = groupingSet;
        end
    end

    %grouping funcs
    set1    = [1, 12, 2; 3, 11, 4; 5, 10, 6; 7, 9, 8];
    set2    = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12];
    DNOMA   = DNOMAGrouping(numPoints);
    DNLUPA  = DNLUPAGrouping(numPoints);
    MUG     = MUGGrouping(numPoints);
    LCG     = LCGGrouping(points);
    DEC     = DECGrouping(points);
    greedyTriplets = greedyGrouping(points, weights);

    %q learning params, print progress
    qParams = struct( ...
        'episodes',      1000, ...   %higher = more accurate
        'alpha',         0.35, ...   %learning rate
        'gamma',         0.92, ...   %discount
        'epsilon_start', 0.90, ...   %explore start
        'epsilon_end',   0.05, ...   %expl final
        'epsilon_decay', 0.995, ...  %decay per ep
        'log_every',     100, ...     %print every x eps
        'verbose',       true ...    %print switch
    );
    [qTriplets, qScore] = qLearningGrouping(points, weights, qParams);

    %comment out the ones u dont want to plot
    plotGroupings(points, bestTriplets, 'Brute Force', weights);
    % plotGroupings(points, set1,        'Set 1',        weights);
    % plotGroupings(points, set2,        'Set 2',        weights);
    % plotGroupings(points, DNOMA,       'DNOMA',        weights);
    % plotGroupings(points, DNLUPA,      'DNLUPA',       weights);
    % plotGroupings(points, MUG,         'MUG',          weights);
    % plotGroupings(points, LCG,         'LCG',          weights);
    % plotGroupings(points, DEC,         'DEC',          weights);
    plotGroupings(points, greedyTriplets, 'Greedy',    weights);
    plotGroupings(points, qTriplets,   'Q-Learning',   weights);

    %comment out the results u dont want to print out
    results.BruteForce = struct('triplets', bestTriplets, ...
        'weights', getWeight(bestTriplets, weights), ...
        'totalScore', maxScore, 'complexity', 'O(n^n)');

    results.set1 = struct('triplets', set1, ...
        'weights', getWeight(set1, weights), ...
        'totalScore', calcScore(set1, points, weights), 'complexity', 'O(1)');

    results.set2 = struct('triplets', set2, ...
        'weights', getWeight(set2, weights), ...
        'totalScore', calcScore(set2, points, weights), 'complexity', 'O(1)');

    results.DNOMA = struct('triplets', DNOMA, ...
        'weights', getWeight(DNOMA, weights), ...
        'totalScore', calcScore(DNOMA, points, weights), 'complexity', 'O(nlogn)');

    results.DNLUPA = struct('triplets', DNLUPA, ...
        'weights', getWeight(DNLUPA, weights), ...
        'totalScore', calcScore(DNLUPA, points, weights), 'complexity', 'O(nlogn)');

    results.MUG = struct('triplets', MUG, ...
        'weights', getWeight(MUG, weights), ...
        'totalScore', calcScore(MUG, points, weights), 'complexity', 'O(n)');

    results.LCG = struct('triplets', LCG, ...
        'weights', getWeight(LCG, weights), ...
        'totalScore', calcScore(LCG, points, weights), 'complexity', 'O(nlogn)');

    results.DEC = struct('triplets', DEC, ...
        'weights', getWeight(DEC, weights), ...
        'totalScore', calcScore(DEC, points, weights), 'complexity', 'O(nlogn)');

    results.Greedy = struct('triplets', greedyTriplets, ...
        'weights', getWeight(greedyTriplets, weights), ...
        'totalScore', calcScore(greedyTriplets, points, weights), 'complexity', 'O(n^3)');

    results.QLearning = struct('triplets', qTriplets, ...
        'weights', getWeight(qTriplets, weights), ...
        'totalScore', qScore, 'complexity', 'O(E·n^2)'); % E = episodes

    printTripletResults(results);
    compareBruteForce(results);
end

%rest of these r helpers

function triplets = DNOMAGrouping(N)
    g1 = 1:4; g2 = 5:8; g3 = 9:12;
    triplets = [g1(1), g2(1), g3(1);
                g1(2), g2(2), g3(2);
                g1(3), g2(3), g3(3);
                g1(4), g2(4), g3(4)];
end

function triplets = DNLUPAGrouping(~)
    triplets = [1,6,12; 2,5,11; 3,8,10; 4,7,9];
end

function triplets = MUGGrouping(N)
    triplets = reshape(1:N, 3, [])';
end

function triplets = LCGGrouping(points)
    [~, order] = sort(sqrt(sum(points.^2, 2)));
    triplets = reshape(order, 3, [])';
end

function triplets = DECGrouping(points)
    strengths = sum(points, 2);
    [~, sorted] = sort(strengths, 'descend');
    triplets = reshape(sorted, 3, [])';
end

function triplets = greedyGrouping(points, weights)
    indices = 1:size(points,1);
    triplets = [];
    eta = 3;
    alpha1 = 0.6; alpha2 = 0.3; alpha3 = 0.1;
    P = 1; N0 = 1e-4;

    while length(indices) >= 3
        bestScore = -inf; bestGroup = [];
        for i = 1:length(indices)
            for j = i+1:length(indices)
                for k = j+1:length(indices)
                    idx = [indices(i), indices(j), indices(k)];
                    d = vecnorm(points(idx, :)');             % per-user |h| based on distance
                    z = raylrnd(1, [1,3]);                    % Rayleigh fading
                    h = (1 ./ d.^(eta/2)) .* z;               % pathloss*fading
                    h = sort(h, 'descend');
                    R1 = log2(1 + (alpha1 * P * h(1)^2) / (alpha2 * P * h(1)^2 + alpha3 * P * h(1)^2 + N0));
                    R2 = log2(1 + (alpha2 * P * h(2)^2) / (alpha3 * P * h(2)^2 + N0));
                    R3 = log2(1 + (alpha3 * P * h(3)^2) / N0);
                    U  = R1 + R2 + R3;
                    w  = mean(weights(idx));
                    sc = U * w;
                    if sc > bestScore
                        bestScore = sc;
                        bestGroup = idx;
                    end
                end
            end
        end
        triplets = [triplets; bestGroup];
        indices = setdiff(indices, bestGroup);
    end
end

function score = calcScore(groupingSet, points, weights)
    alpha1 = 0.6; alpha2 = 0.3; alpha3 = 0.1;
    P = 1; N0 = 1e-4; eta = 3;
    score = 0;
    for i = 1:size(groupingSet, 1)
        idx = groupingSet(i, :);
        d = vecnorm(points(idx, :)');
        z = raylrnd(1, [1,3]);
        h = (1 ./ d.^(eta/2)) .* z;
        h = sort(h, 'descend');
        R1 = log2(1 + (alpha1 * P * h(1)^2) / (alpha2 * P * h(1)^2 + alpha3 * P * h(1)^2 + N0));
        R2 = log2(1 + (alpha2 * P * h(2)^2) / (alpha3 * P * h(2)^2 + N0));
        R3 = log2(1 + (alpha3 * P * h(3)^2) / N0);
        U  = R1 + R2 + R3;
        w  = mean(weights(idx));
        score = score + U * w;
    end
end

function groupings = makeGroupings(indices)
    if isempty(indices)
        groupings = {[]};
        return;
    end
    groupings = {};
    first = indices(1);
    for i = 2:length(indices)
        for j = i+1:length(indices)
            rest = indices([2:i-1, i+1:j-1, j+1:end]);
            sub = makeGroupings(rest);
            for k = 1:length(sub)
                groupings{end+1} = [first, indices(i), indices(j); sub{k}]; %#ok<AGROW>
            end
        end
    end
end

function plotGroupings(points, triplets, name, weights)
    figure('Color', 'w');
    scatter3(points(:,1), points(:,2), points(:,3), 100, 'filled'); hold on;
    colors = lines(size(triplets,1));
    for i = 1:size(triplets,1)
        idx = triplets(i,:);
        c = colors(i,:);
        plot3(points(idx,1), points(idx,2), points(idx,3), '-', 'LineWidth', 2, 'Color', c);
        plot3([points(idx(1),1), points(idx(3),1)], ...
              [points(idx(1),2), points(idx(3),2)], ...
              [points(idx(1),3), points(idx(3),3)], '-', 'LineWidth', 2, 'Color', c);
    end
    for i = 1:size(points,1)
        text(points(i,1), points(i,2), points(i,3), ...
            sprintf('%.2f', weights(i)), ...
            'FontSize', 10, 'FontWeight', 'bold', ...
            'Color', 'w', 'EdgeColor', 'k', ...
            'BackgroundColor', 'k', 'Margin', 1, ...
            'HorizontalAlignment', 'left');
    end
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title([name, ' Triplets (Weighted)']);
    grid on;
end

function weightsMatrix = getWeight(triplets, weights)
    weightsMatrix = zeros(size(triplets));
    for i = 1:size(triplets, 1)
        weightsMatrix(i, :) = weights(triplets(i, :));
    end
end

function printTripletResults(results)
    fields = fieldnames(results);
    for i = 1:length(fields)
        key = fields{i};
        s = results.(key);
        trip = s.triplets;
        fprintf('\n%s Grouping (%s):\n', key, s.complexity);
        disp(trip);
        fprintf('Triplet Weights:\n');
        for j = 1:size(trip,1)
            w = s.weights(j,:);
            fprintf('  Triplet %d: [%.2f, %.2f, %.2f]\n', j, w(1), w(2), w(3));
        end
        fprintf('Total Score (Weighted Utility): %.4f\n\n', s.totalScore);
    end
end

function compareBruteForce(results)
    bruteForceScore = results.BruteForce.totalScore;
    methods = fieldnames(results);
    disp(' ');
    for i = 1:length(methods)
        method = methods{i};
        if strcmp(method, 'BruteForce')
            continue;
        end
        delta = bruteForceScore - results.(method).totalScore;
        disp(['Brute Force v ', method, '? ', num2str(delta)]);
    end
    disp('The higher the difference, the more optimal brute force is for (weighted utility).');
end

%q learning logic

function [triplets, episodeBestScore] = qLearningGrouping(points, weights, p)
    %logging results
    if ~isfield(p,'log_every'), p.log_every = max(1, floor(p.episodes/20)); end %log every 5%
    if ~isfield(p,'verbose'),   p.verbose   = true; end

    %STATE = remaining users unassigned + current partial triplet (orderless)
    %ACTION = choose one index from remaining to add to current triplet
    %REWARD = weighted utility Uw when a triplet completes else 0
    N = size(points,1);

    %q table map: key = state | a -> value
    Q = containers.Map('KeyType','char','ValueType','double');

    episodeBestScore = -inf;
    episodeBestTrip  = [];

    eps = p.epsilon_start;
    for ep = 1:p.episodes
        remaining = true(1,N);
        current   = [];      %index for curr unfinished trip
        cumScore  = 0;
        traceTrip = [];      %final trip of episode

        %episode
        while any(remaining)
            sKey = makeStateKey(remaining, current);

            actions = find(remaining);
            %explore or exploit
            if rand < eps
                a = actions(randi(length(actions)));
            else
                a = argmaxQ(Q, sKey, actions);
            end
            %maintain env
            remaining(a) = false;
            current = [current, a];
            reward = 0;

            if numel(current) == 3
                %get trip utility
                reward = tripletUtility(current, points, weights);
                cumScore = cumScore + reward;
                traceTrip = [traceTrip; current];
                current = []; %reset for next
            end

            %next state, maxQ
            sNextKey = makeStateKey(remaining, current);
            nextActions = find(remaining);
            if isempty(nextActions)
                maxQnext = 0;
            else
                maxQnext = maxQ(Q, sNextKey, nextActions);
            end

            %update q
            qaKey = makeQAKey(sKey, a);
            oldQ  = 0;
            if isKey(Q, qaKey), oldQ = Q(qaKey); end
            Q(qaKey) = (1 - p.alpha)*oldQ + p.alpha*(reward + p.gamma*maxQnext);
        end

        %track best episode score
        if cumScore > episodeBestScore
            episodeBestScore = cumScore;
            episodeBestTrip  = traceTrip;
        end

        %less exploration over time
        eps = max(p.epsilon_end, eps * p.epsilon_decay);

        %log progress
        if p.verbose && (ep == 1 || ep == p.episodes || mod(ep, p.log_every) == 0)
            fprintf('Episode %4d/%4d | episode utility = %.4f | best = %.4f\n', ...
                ep, p.episodes, cumScore, episodeBestScore);
            drawnow limitrate
        end
    end

    %greedy policy from learned q
    remaining = true(1,N);
    current   = [];
    triplets  = [];
    while any(remaining)
        sKey = makeStateKey(remaining, current);
        actions = find(remaining);
        a = argmaxQ(Q, sKey, actions);
        remaining(a) = false;
        current = [current, a];
        if numel(current) == 3
            triplets = [triplets; current];
            current = [];
        end
    end

    %invalid side the use best case
    if isempty(triplets) || size(triplets,2) ~= 3 || size(triplets,1) ~= N/3
        triplets = episodeBestTrip;
    end
end

%orderless current, shrink state space, reusable q table entires
function key = makeStateKey(remaining, current)
    %remaining to bits
    r = char('0' + remaining);
    if isempty(current)
        c = '[]';
    else
        c = sprintf('[%s]', sprintf('%d,', sort(current))); % orderless
        if c(end)==',', c(end)=[]; end
    end
    key = [r,'|',c];
end

%append aciton to state key
function k = makeQAKey(sKey, a)
    k = [sKey,'|a:',num2str(a)];
end

%available actions, pick action with highest stored q val
%0 for unseen keys
function a = argmaxQ(Q, sKey, actions)
    bestVal = -inf; a = actions(1);
    for i = 1:length(actions)
        k = makeQAKey(sKey, actions(i));
        v = 0; if isKey(Q, k), v = Q(k); end
        if v > bestVal
            bestVal = v; a = actions(i);
        end
    end
end

%return max_a Q(s,a) else return 0 if no actions
function m = maxQ(Q, sKey, actions)
    vals = zeros(1,length(actions));
    for i = 1:length(actions)
        k = makeQAKey(sKey, actions(i));
        if isKey(Q, k), vals(i) = Q(k); end
    end
    if isempty(vals), m = 0; else, m = max(vals); end
end

%what we had before
function rew = tripletUtility(idx, points, weights)
    alpha1 = 0.6; alpha2 = 0.3; alpha3 = 0.1;
    P = 1; N0 = 1e-4; eta = 3;

    d = vecnorm(points(idx, :)');
    z = raylrnd(1, [1,3]);
    h = (1 ./ d.^(eta/2)) .* z;
    h = sort(h, 'descend');
    R1 = log2(1 + (alpha1 * P * h(1)^2) / (alpha2 * P * h(1)^2 + alpha3 * P * h(1)^2 + N0));
    R2 = log2(1 + (alpha2 * P * h(2)^2) / (alpha3 * P * h(2)^2 + N0));
    R3 = log2(1 + (alpha3 * P * h(3)^2) / N0);
    U  = R1 + R2 + R3;
    w  = mean(weights(idx));
    rew = U * w;
end

%local raylrnd
function r = raylrnd(b, sz)
    if numel(sz) == 1, sz = [1, sz]; end
    u = max(eps, rand(sz));
    r = b .* sqrt(-2 .* log(u));
end
