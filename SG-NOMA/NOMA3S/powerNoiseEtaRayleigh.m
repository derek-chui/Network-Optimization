function powerNoiseEtaRayleigh()

    [P_vals, bruteP, greedyP, dnomaP, lcgP, jvP]   = getPowerUtilityData();
    [N0_vals, bruteN, greedyN, dnomaN, lcgN, jvN]   = getNoiseUtilityData();
    [eta_vals, bruteE, greedyE, dnomaE, lcgE, jvE]   = getEtaUtilityData();
    [sigma_vals, bruteS, greedyS, dnomaS, lcgS, jvS] = getRayleighUtilityData();

    figW = 7.5;
    figH = 8.5;
    fig  = figure('Color','w','Units','inches','Position',[0.5 0.5 figW figH], ...
                  'Renderer','painters');

    t = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    ax1 = nexttile;
    plot(P_vals, bruteP, '-o', 'LineWidth', 1.8, 'DisplayName', 'Brute Force'); hold on;
    plot(P_vals, greedyP, '-s', 'LineWidth', 1.8, 'DisplayName', 'SG-NOMA');
    plot(P_vals, dnomaP, '-^', 'LineWidth', 1.8, 'DisplayName', 'DNOMA');
    plot(P_vals, lcgP, '-d', 'LineWidth', 1.8, 'DisplayName', 'LCG');
    plot(P_vals, jvP, '-p', 'LineWidth', 1.8, 'DisplayName', 'JV'); hold off;
    grid on; box on;
    xlabel('Total Transmission Power (P)'); ylabel('Total Utility');
    title('Utility vs. Transmission Power P');
    legend('show','Location','northwest');

    ax2 = nexttile;
    semilogx(N0_vals, bruteN, '-o', 'LineWidth', 1.8, 'DisplayName', 'Brute Force'); hold on;
    semilogx(N0_vals, greedyN, '-s', 'LineWidth', 1.8, 'DisplayName', 'SG-NOMA');
    semilogx(N0_vals, dnomaN, '-^', 'LineWidth', 1.8, 'DisplayName', 'DNOMA');
    semilogx(N0_vals, lcgN, '-d', 'LineWidth', 1.8, 'DisplayName', 'LCG');
    semilogx(N0_vals, jvN, '-p', 'LineWidth', 1.8, 'DisplayName', 'JV'); hold off;
    grid on; box on;
    xlabel('Noise Power (N_0)','Interpreter','tex'); ylabel('Total Utility');
    title('Utility vs. Noise Power N_0');
    legend('show','Location','southwest');

    ax3 = nexttile;
    plot(eta_vals, bruteE, '-o', 'LineWidth', 1.8, 'DisplayName', 'Brute Force'); hold on;
    plot(eta_vals, greedyE, '-s', 'LineWidth', 1.8, 'DisplayName', 'SG-NOMA');
    plot(eta_vals, dnomaE, '-^', 'LineWidth', 1.8, 'DisplayName', 'DNOMA');
    plot(eta_vals, lcgE, '-d', 'LineWidth', 1.8, 'DisplayName', 'LCG');
    plot(eta_vals, jvE, '-p', 'LineWidth', 1.8, 'DisplayName', 'JV'); hold off;
    grid on; box on;
    xlabel('Path Loss Exponent (\eta)','Interpreter','tex'); ylabel('Total Utility');
    title('Utility vs. Path Loss Exponent \eta');
    legend('show','Location','northeast');

    ax4 = nexttile;
    plot(sigma_vals, bruteS, '-o', 'LineWidth', 1.8, 'DisplayName', 'Brute Force'); hold on;
    plot(sigma_vals, greedyS, '-s', 'LineWidth', 1.8, 'DisplayName', 'SG-NOMA');
    plot(sigma_vals, dnomaS, '-^', 'LineWidth', 1.8, 'DisplayName', 'DNOMA');
    plot(sigma_vals, lcgS, '-d', 'LineWidth', 1.8, 'DisplayName', 'LCG');
    plot(sigma_vals, jvS, '-p', 'LineWidth', 1.8, 'DisplayName', 'JV'); hold off;
    grid on; box on;
    xlabel('Rayleigh Scale Parameter (\sigma)','Interpreter','tex'); ylabel('Total Utility');
    title('Utility vs. Rayleigh Fading Parameter \sigma');
    legend('show','Location','northwest');

    set(fig, 'PaperUnits','inches');
    set(fig, 'PaperSize', [figW figH]);
    set(fig, 'PaperPosition', [0 0 figW figH]);
    set(fig, 'PaperPositionMode','manual');
    orient(fig,'portrait');

    print(fig, '-dpdf',  '-painters', 'powerNoiseEtaRayleigh.pdf');
    print(fig, '-depsc', '-painters', '-loose', 'powerNoiseEtaRayleigh.eps');
    try
        print(fig, '-dsvg', '-painters', 'powerNoiseEtaRayleigh.svg');
    catch
    end

    saveTile(ax1, 'powerNoiseEtaRayleigh_P');
    saveTile(ax2, 'powerNoiseEtaRayleigh_N0');
    saveTile(ax3, 'powerNoiseEtaRayleigh_eta');
    saveTile(ax4, 'powerNoiseEtaRayleigh_sigma');

    function saveTile(ax, base)
        f = figure('Visible','off','Units','inches','Position',[0.5 0.5 6 4], 'Renderer','painters');
        axCopy = copyobj(ax, f);
        set(axCopy,'Units','normalized','Position',[0.13 0.14 0.84 0.79]);
        set(f,'PaperUnits','inches','PaperSize',[6 4], 'PaperPosition',[0 0 6 4], 'PaperPositionMode','manual');
        print(f, '-dpdf',  '-painters', [base '.pdf']);
        print(f, '-depsc', '-painters', '-loose', [base '.eps']);
        try
            print(f, '-dsvg', '-painters', [base '.svg']);
        catch
        end
        close(f);
    end

    function [P_vals, brute, greedy, dnoma, lcg, jv] = getPowerUtilityData()
        P_vals = [0.1, 0.5, 1, 2, 5];
        numPoints = 12; eta = 3; alpha = 0.6; N0 = 1e-4; sigma = 1;
        brute = zeros(size(P_vals)); greedy = brute; dnoma = brute; lcg = brute; jv = brute;
        for k = 1:length(P_vals)
            P = P_vals(k);
            rng(42); [points, r] = generateScenario(numPoints, 10,10,10, sigma);
            pairings = makePairings(1:numPoints);
            maxU = -inf;
            for i = 1:length(pairings)
                u = calcUtility(pairings{i}, points, r, eta, alpha, P, N0);
                if u > maxU, maxU = u; end
            end
            brute(k)  = maxU;
            greedy(k) = calcUtility(greedyPairing(points, r, eta, alpha, P, N0), points, r, eta, alpha, P, N0);
            dnoma(k)  = calcUtility(dnomaPairs(points), points, r, eta, alpha, P, N0);
            lcg(k) = calcUtility(lcgPairs(points), points, r, eta, alpha, P, N0);
            jv(k) = calcUtility(jvPairs(points, r, eta, alpha, P, N0), points, r, eta, alpha, P, N0);
        end
    end

    function [N0_vals, brute, greedy, dnoma, lcg, jv] = getNoiseUtilityData()
        N0_vals = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2];
        numPoints = 12; eta = 3; alpha = 0.6; P = 1; sigma = 1;
        brute = zeros(size(N0_vals)); greedy = brute; dnoma = brute; lcg = brute; jv = brute;
        for k = 1:length(N0_vals)
            N0 = N0_vals(k);
            rng(42); [points, r] = generateScenario(numPoints, 10,10,10, sigma);
            pairings = makePairings(1:numPoints);
            maxU = -inf;
            for i = 1:length(pairings)
                u = calcUtility(pairings{i}, points, r, eta, alpha, P, N0);
                if u > maxU, maxU = u; end
            end
            brute(k) = maxU;
            greedy(k) = calcUtility(greedyPairing(points, r, eta, alpha, P, N0), points, r, eta, alpha, P, N0);
            dnoma(k) = calcUtility(dnomaPairs(points), points, r, eta, alpha, P, N0);
            lcg(k) = calcUtility(lcgPairs(points), points, r, eta, alpha, P, N0);
            jv(k) = calcUtility(jvPairs(points, r, eta, alpha, P, N0), points, r, eta, alpha, P, N0);
        end
    end

    function [eta_vals, brute, greedy, dnoma, lcg, jv] = getEtaUtilityData()
        eta_vals = 2:6;
        numPoints = 12; alpha = 0.6; P = 1; N0 = 1e-4; sigma = 1;
        brute = zeros(size(eta_vals)); greedy = brute; dnoma = brute; lcg = brute; jv = brute;
        for k = 1:length(eta_vals)
            eta = eta_vals(k);
            rng(42); [points, r] = generateScenario(numPoints, 10,10,10, sigma);
            pairings = makePairings(1:numPoints);
            maxU = -inf;
            for i = 1:length(pairings)
                u = calcUtility(pairings{i}, points, r, eta, alpha, P, N0);
                if u > maxU, maxU = u; end
            end
            brute(k) = maxU;
            greedy(k) = calcUtility(greedyPairing(points, r, eta, alpha, P, N0), points, r, eta, alpha, P, N0);
            dnoma(k) = calcUtility(dnomaPairs(points), points, r, eta, alpha, P, N0);
            lcg(k) = calcUtility(lcgPairs(points), points, r, eta, alpha, P, N0);
            jv(k) = calcUtility(jvPairs(points, r, eta, alpha, P, N0), points, r, eta, alpha, P, N0);
        end
    end

    function [sigma_vals, brute, greedy, dnoma, lcg, jv] = getRayleighUtilityData()
        sigma_vals = [0.5, 1.0, 1.5, 2.0, 2.5];
        numPoints = 12; eta = 3; alpha = 0.6; P = 1; N0 = 1e-4;
        brute = zeros(size(sigma_vals)); greedy = brute; dnoma = brute; lcg = brute; jv = brute;
        for k = 1:length(sigma_vals)
            sigma = sigma_vals(k);
            rng(42); [points, r] = generateScenario(numPoints, 10,10,10, sigma);
            pairings = makePairings(1:numPoints);
            maxU = -inf;
            for i = 1:length(pairings)
                u = calcUtility(pairings{i}, points, r, eta, alpha, P, N0);
                if u > maxU, maxU = u; end
            end
            brute(k) = maxU;
            greedy(k) = calcUtility(greedyPairing(points, r, eta, alpha, P, N0), points, r, eta, alpha, P, N0);
            dnoma(k) = calcUtility(dnomaPairs(points), points, r, eta, alpha, P, N0);
            lcg(k) = calcUtility(lcgPairs(points), points, r, eta, alpha, P, N0);
            jv(k) = calcUtility(jvPairs(points, r, eta, alpha, P, N0), points, r, eta, alpha, P, N0);
        end
    end

    function [points, r] = generateScenario(n, xr, yr, zr, sigma)
        x = rand(1,n) * xr; y = rand(1,n) * yr; z = rand(1,n) * zr;
        points = [x; y; z]';
        r = raylrnd(sigma, n, 1);
        dist = sqrt(sum(points.^2, 2));
        [~, idx] = sort(dist);
        points = points(idx, :);
        r = r(idx);
    end

    function U = calcUtility(pairs, points, r, eta, alpha, P, N0)
        U = 0;
        for jj = 1:size(pairs,1)
            i1 = pairs(jj,1); i2 = pairs(jj,2);
            d1 = norm(points(i1,:)); d2 = norm(points(i2,:));
            h1 = (1 / (d1^(eta/2))) * r(i1);
            h2 = (1 / (d2^(eta/2))) * r(i2);
            R1 = log2(1 + (alpha * P * h1^2) / ((1 - alpha) * P * h1^2 + N0));
            R2 = log2(1 + ((1 - alpha) * P * h2^2) / N0);
            U = U + R1 + R2;
        end
    end

    function pairings = makePairings(indices)
        if isempty(indices), pairings = {[]}; return; end
        pairings = {}; first = indices(1);
        for ii = 2:length(indices)
            second = indices(ii);
            remaining = indices([2:ii-1, ii+1:end]);
            subs = makePairings(remaining);
            for jj = 1:length(subs)
                pairings{end+1} = [[first, second]; subs{jj}];
            end
        end
    end

    function pairs = greedyPairing(points, r, eta, alpha, P, N0)
        N = size(points,1); remaining = 1:N; pairs = [];
        while numel(remaining) > 1
            bestU = -inf; bestPair = [];
            for ii = 1:numel(remaining)
                for jj = ii+1:numel(remaining)
                    pr = [remaining(ii), remaining(jj)];
                    u = calcUtility(pr, points, r, eta, alpha, P, N0);
                    if u > bestU, bestU = u; bestPair = pr; end
                end
            end
            pairs = [pairs; bestPair]; %#ok<AGROW>
            remaining = setdiff(remaining, bestPair);
        end
    end

    function pairs = dnomaPairs(points)
        N = size(points,1); idx = 1:N;
        G1 = idx(1:N/4); G2 = idx(N/4+1:N/2);
        G3 = idx(N/2+1:3*N/4); G4 = idx(3*N/4+1:end);
        pairs = [G1(:), G3(:); G2(:), G4(:)];
    end

    function pairs = lcgPairs(points)
        N = size(points,1); remaining = 1:N; pairs = [];
        while numel(remaining) > 1
            bestDist = inf; bestPair = [];
            for ii = 1:numel(remaining)
                for jj = ii+1:numel(remaining)
                    d = norm(points(remaining(ii),:) - points(remaining(jj),:));
                    if d < bestDist, bestDist = d; bestPair = [remaining(ii), remaining(jj)]; end
                end
            end
            pairs = [pairs; bestPair];
            remaining = setdiff(remaining, bestPair);
        end
    end

    function pairs = jvPairs(points, r, eta, alpha, P, N0)
        N = size(points,1);
        close = 1:(N/2); far = (N/2+1):N;
        cost = zeros(length(close), length(far));
        for ii = 1:length(close)
            for jj = 1:length(far)
                pr = [close(ii), far(jj)];
                cost(ii,jj) = -calcUtility(pr, points, r, eta, alpha, P, N0);
            end
        end
        [match, ~] = matchpairs(cost, -1);
        pairs = [close(match(:,1))', far(match(:,2))'];
    end
end
