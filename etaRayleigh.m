function etaRayleigh()
    [eta_vals, brute, greedy, dnoma, lcg, jv] = getEtaUtilityData();
    [sigma_vals, bruteS, greedyS, dnomaS, lcgS, jvS] = getRayleighUtilityData();

    figure('Color', 'w', 'Position', [100 100 700 800]);

    subplot(2,1,1);
    plot(eta_vals, brute, '-o', 'LineWidth', 2, 'DisplayName', 'Brute Force'); hold on;
    plot(eta_vals, greedy, '-s', 'LineWidth', 2, 'DisplayName', 'SG-NOMA');
    plot(eta_vals, dnoma,  '-^', 'LineWidth', 2, 'DisplayName', 'DNOMA');
    plot(eta_vals, lcg,    '-d', 'LineWidth', 2, 'DisplayName', 'LCG');
    plot(eta_vals, jv,     '-p', 'LineWidth', 2, 'DisplayName', 'JV'); hold off;
    xlabel('Path Loss Exponent (\eta)');
    ylabel('Total Utility');
    title('Utility vs. Path Loss Exponent \eta by Algorithm');
    legend('show', 'Location', 'northeast');
    grid on;

    subplot(2,1,2);
    plot(sigma_vals, bruteS, '-o', 'LineWidth', 2, 'DisplayName', 'Brute Force'); hold on;
    plot(sigma_vals, greedyS, '-s', 'LineWidth', 2, 'DisplayName', 'SG-NOMA');
    plot(sigma_vals, dnomaS,  '-^', 'LineWidth', 2, 'DisplayName', 'DNOMA');
    plot(sigma_vals, lcgS,    '-d', 'LineWidth', 2, 'DisplayName', 'LCG');
    plot(sigma_vals, jvS,     '-p', 'LineWidth', 2, 'DisplayName', 'JV'); hold off;
    xlabel('Rayleigh Scale Parameter (\sigma)');
    ylabel('Total Utility');
    title('Utility vs. Rayleigh Fading Parameter \sigma by Algorithm');
    legend('show', 'Location', 'northwest');
    grid on;
end

function [eta_values, brute, greedy, dnoma, lcg, jv] = getEtaUtilityData()
    eta_values = 2:6;
    brute = zeros(size(eta_values));
    greedy = zeros(size(eta_values));
    dnoma = zeros(size(eta_values));
    lcg = zeros(size(eta_values));
    jv = zeros(size(eta_values));

    for k = 1:length(eta_values)
        eta = eta_values(k);
        rng(42);
        [points, r] = generateScenario(12, 10, 10, 10);

        brute(k) = runBruteForce(points, r, eta);
        greedy(k) = runGreedy(points, r, eta);
        dnoma(k) = runDNOMA(points, r, eta);
        lcg(k) = runLCG(points, r, eta);
        jv(k) = runJV(points, r, eta);
    end
end

function [sigma_vals, brute, greedy, dnoma, lcg, jv] = getRayleighUtilityData()
    sigma_vals = [0.5, 1.0, 1.5, 2.0, 2.5];
    brute = zeros(size(sigma_vals));
    greedy = zeros(size(sigma_vals));
    dnoma = zeros(size(sigma_vals));
    lcg = zeros(size(sigma_vals));
    jv = zeros(size(sigma_vals));
    eta = 3;

    for k = 1:length(sigma_vals)
        sigma = sigma_vals(k);
        rng(42);
        [points, r] = generateScenarioWithSigma(12, 10, 10, 10, sigma);

        pairings = makePairings(1:12);
        maxUtil = -inf;
        for i = 1:length(pairings)
            u = calcUtility(pairings{i}, points, r, eta, 0.6, 1, 1e-4);
            if u > maxUtil
                maxUtil = u;
            end
        end
        brute(k) = maxUtil;
        greedy(k) = calcUtility(greedyPairing(points, r, eta), points, r, eta, 0.6, 1, 1e-4);
        dnoma(k)  = calcUtility(dnomaPairs(points), points, r, eta, 0.6, 1, 1e-4);
        lcg(k)    = calcUtility(lcgPairs(points), points, r, eta, 0.6, 1, 1e-4);
        jv(k)     = calcUtility(jvPairs(points, r, eta), points, r, eta, 0.6, 1, 1e-4);
    end
end

function utility = runBruteForce(points, r, eta)
    N = size(points,1);
    allPairings = makePairings(1:N);
    maxUtility = -inf;
    for i = 1:length(allPairings)
        u = calcUtility(allPairings{i}, points, r, eta, 0.6, 1, 1e-4);
        if u > maxUtility
            maxUtility = u;
        end
    end
    utility = maxUtility;
end

function utility = runGreedy(points, r, eta)
    pairs = greedyPairing(points, r, eta);
    utility = calcUtility(pairs, points, r, eta, 0.6, 1, 1e-4);
end

function utility = runDNOMA(points, r, eta)
    pairs = dnomaPairs(points);
    utility = calcUtility(pairs, points, r, eta, 0.6, 1, 1e-4);
end

function utility = runLCG(points, r, eta)
    pairs = lcgPairs(points);
    utility = calcUtility(pairs, points, r, eta, 0.6, 1, 1e-4);
end

function utility = runJV(points, r, eta)
    pairs = jvPairs(points, r, eta);
    utility = calcUtility(pairs, points, r, eta, 0.6, 1, 1e-4);
end

function [points, r] = generateScenario(n, xr, yr, zr)
    x = rand(1, n) * xr;
    y = rand(1, n) * yr;
    z = rand(1, n) * zr;
    points = [x; y; z]';
    r = raylrnd(1, n, 1);
    dist = sqrt(sum(points.^2, 2));
    [~, idx] = sort(dist);
    points = points(idx, :);
    r = r(idx);
end

function [points, r] = generateScenarioWithSigma(n, xr, yr, zr, sigma)
    x = rand(1, n) * xr;
    y = rand(1, n) * yr;
    z = rand(1, n) * zr;
    points = [x; y; z]';
    r = raylrnd(sigma, n, 1);
    dist = sqrt(sum(points.^2, 2));
    [~, idx] = sort(dist);
    points = points(idx, :);
    r = r(idx);
end

function U = calcUtility(pairs, points, r, eta, alpha, P, N0)
    U = 0;
    for j = 1:size(pairs, 1)
        i1 = pairs(j,1); i2 = pairs(j,2);
        d1 = norm(points(i1,:));
        d2 = norm(points(i2,:));
        h1 = (1 / (d1^(eta/2))) * r(i1);
        h2 = (1 / (d2^(eta/2))) * r(i2);
        R1 = log2(1 + (alpha * P * h1^2) / ((1 - alpha) * P * h1^2 + N0));
        R2 = log2(1 + ((1 - alpha) * P * h2^2) / N0);
        U = U + R1 + R2;
    end
end

function pairings = makePairings(indices)
    if isempty(indices)
        pairings = {[]}; return;
    end
    pairings = {};
    first = indices(1);
    for i = 2:length(indices)
        second = indices(i);
        rest = indices([2:i-1, i+1:end]);
        subs = makePairings(rest);
        for j = 1:length(subs)
            pairings{end+1} = [[first, second]; subs{j}];
        end
    end
end

function pairs = greedyPairing(points, r, eta)
    N = size(points, 1); remaining = 1:N; pairs = [];
    while numel(remaining) > 1
        bestU = -inf; bestPair = [];
        for i = 1:numel(remaining)
            for j = i+1:numel(remaining)
                pair = [remaining(i), remaining(j)];
                U = calcUtility(pair, points, r, eta, 0.6, 1, 1e-4);
                if U > bestU
                    bestU = U;
                    bestPair = pair;
                end
            end
        end
        pairs = [pairs; bestPair];
        remaining = setdiff(remaining, bestPair);
    end
end

function pairs = dnomaPairs(points)
    N = size(points, 1); idx = 1:N;
    G1 = idx(1:N/4); G2 = idx(N/4+1:N/2);
    G3 = idx(N/2+1:3*N/4); G4 = idx(3*N/4+1:end);
    pairs = [G1(:), G3(:); G2(:), G4(:)];
end

function pairs = lcgPairs(points)
    N = size(points,1); remaining = 1:N; pairs = [];
    while numel(remaining) > 1
        bestDist = inf; bestPair = [];
        for i = 1:numel(remaining)
            for j = i+1:numel(remaining)
                d = norm(points(remaining(i),:) - points(remaining(j),:));
                if d < bestDist
                    bestDist = d;
                    bestPair = [remaining(i), remaining(j)];
                end
            end
        end
        pairs = [pairs; bestPair];
        remaining = setdiff(remaining, bestPair);
    end
end

function pairs = jvPairs(points, r, eta)
    N = size(points, 1);
    close = 1:(N/2); far = (N/2+1):N;
    costMatrix = zeros(length(close), length(far));
    for i = 1:length(close)
        for j = 1:length(far)
            pair = [close(i), far(j)];
            costMatrix(i,j) = -calcUtility(pair, points, r, eta, 0.6, 1, 1e-4);
        end
    end
    [match, ~] = matchpairs(costMatrix, -1);
    pairs = [close(match(:,1))', far(match(:,2))'];
end
