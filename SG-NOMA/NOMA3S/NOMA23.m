function NOMA23()
    rng(42);
    numPoints = 12;
    halfRange = 5;
    x = (rand(1, numPoints) - 0.5) * 2 * halfRange;
    y = (rand(1, numPoints) - 0.5) * 2 * halfRange;
    z = (rand(1, numPoints) - 0.5) * 2 * halfRange;
    points = [x; y; z]';
    weights = rand(1, numPoints);
    r = raylrnd(1, numPoints, 1);

    dist = sqrt(sum(points.^2, 2));
    [~, idx] = sort(dist);
    points  = points(idx, :);
    weights = weights(idx);
    r = r(idx);

    greedyPairs = greedyPairing(points, r, weights);
    greedyTriplets = greedyGrouping(points, weights);

    fig = figure('Color','w','Position',[100 100 1100 520],'Renderer','painters');
    t = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

    ax1 = nexttile;
    plotPairsStyled(ax1, points, greedyPairs, weights, 'Greedy Pairing (Weighted)', halfRange);

    ax2 = nexttile;
    plotTripletsStyled(ax2, points, greedyTriplets, weights, 'Greedy Triplets (Weighted)', halfRange);

    set(fig,'PaperUnits','inches','InvertHardcopy','off');
    orient(fig,'landscape');
    print(fig, '-dpdf',  '-vector', '-bestfit', 'NOMA23.pdf');

    set(fig,'PaperPositionMode','auto');
    print(fig, '-depsc', '-vector', '-loose', 'NOMA23.eps');

    try
        print(fig, '-dsvg', '-vector', 'NOMA23.svg');
    catch
    end

    exportgraphics(ax1,'NOMA23_pairs.pdf',    'ContentType','vector');
    exportgraphics(ax2,'NOMA23_triplets.pdf', 'ContentType','vector');

    try
        f1 = figure('Visible','off','Renderer','painters');
        ax1copy = copyobj(ax1,f1); set(ax1copy,'Units','normalized','Position',[0.13 0.11 0.775 0.815]);
        print(f1,'-dsvg','-vector','NOMA23_pairs.svg'); close(f1);

        f2 = figure('Visible','off','Renderer','painters');
        ax2copy = copyobj(ax2,f2); set(ax2copy,'Units','normalized','Position',[0.13 0.11 0.775 0.815]);
        print(f2,'-dsvg','-vector','NOMA23_triplets.svg'); close(f2);
    catch
    end

    function pairs = greedyPairing(points_, r_, weights_)
        eta = 3; alpha = 0.6; P = 1; N0 = 1e-4;
        N = size(points_, 1); remaining = 1:N; pairs = [];
        while numel(remaining) > 1
            bestU = -inf; bestPair = [];
            for ii = 1:numel(remaining)
                for jj = ii+1:numel(remaining)
                    i1 = remaining(ii); i2 = remaining(jj);
                    d1 = norm(points_(i1,:)); d2 = norm(points_(i2,:));
                    h1 = (1 / (d1^(eta/2))) * r_(i1);
                    h2 = (1 / (d2^(eta/2))) * r_(i2);
                    R1 = log2(1 + (alpha * P * h1^2) / ((1 - alpha) * P * h1^2 + N0));
                    R2 = log2(1 + ((1 - alpha) * P * h2^2) / N0);
                    w  = mean([weights_(i1), weights_(i2)]);
                    U  = (R1 + R2) * w;
                    if U > bestU
                        bestU = U; bestPair = [i1, i2];
                    end
                end
            end
            pairs = [pairs; bestPair];
            remaining = setdiff(remaining, bestPair);
        end
    end

    function triplets = greedyGrouping(points_, weights_)
        indices = 1:size(points_,1);
        triplets = [];
        eta = 3;
        alpha1 = 0.6; alpha2 = 0.3; alpha3 = 0.1;
        P = 1; N0 = 1e-4;

        while length(indices) >= 3
            bestScore = -inf; bestGroup = [];
            for a = 1:length(indices)
                for b = a+1:length(indices)
                    for c = b+1:length(indices)
                        idx3 = [indices(a), indices(b), indices(c)];
                        d = vecnorm(points_(idx3, :)');
                        z = raylrnd(1, [1,3]);
                        h = (1 ./ d.^(eta/2)) .* z;
                        h = sort(h, 'descend');
                        R1 = log2(1 + (alpha1 * P * h(1)^2) / (alpha2 * P * h(1)^2 + alpha3 * P * h(1)^2 + N0));
                        R2 = log2(1 + (alpha2 * P * h(2)^2) / (alpha3 * P * h(2)^2 + N0));
                        R3 = log2(1 + (alpha3 * P * h(3)^2) / N0);
                        U  = R1 + R2 + R3;
                        w  = mean(weights_(idx3));
                        score = U * w;
                        if score > bestScore
                            bestScore = score; bestGroup = idx3;
                        end
                    end
                end
            end
            triplets = [triplets; bestGroup];
            indices = setdiff(indices, bestGroup);
        end
    end

    function plotPairsStyled(ax, pts, pairs, wts, ttl, hr)
        axes(ax); cla(ax);
        plot3(pts(:,1), pts(:,2), pts(:,3), 'o', ...
              'LineStyle','none','MarkerSize',8, ...
              'MarkerEdgeColor','k','MarkerFaceColor',[0.25 0.25 0.25]); hold on;

        cols = lines(size(pairs,1));
        for k = 1:size(pairs,1)
            idx = pairs(k,:); c = cols(k,:);
            plot3([pts(idx(1),1), pts(idx(2),1)], ...
                  [pts(idx(1),2), pts(idx(2),2)], ...
                  [pts(idx(1),3), pts(idx(2),3)], ...
                  '-', 'LineWidth', 2, 'Color', c);

            [~, posStrong] = min(vecnorm(pts(idx, :)', 2));
            sIdx = idx(posStrong);
            plot3(pts(sIdx,1), pts(sIdx,2), pts(sIdx,3), 'o', ...
                  'MarkerSize', 10, 'LineWidth', 2, ...
                  'MarkerEdgeColor','g','MarkerFaceColor','g');
        end

        labelWeights(pts, wts);
        plot3(0, 0, 0, 'p', 'MarkerSize', 12, 'MarkerEdgeColor','r','MarkerFaceColor','r');
        text(0, 0, 0, ' BS (0,0,0)', 'FontWeight','bold');

        xlim([-hr hr]); ylim([-hr hr]); zlim([-hr hr]);
        axis equal; grid on; view(45,25);
        set(ax,'Projection','orthographic','Layer','top');
        xlabel('X'); ylabel('Y'); zlabel('Z'); title(ttl);
    end

    function plotTripletsStyled(ax, pts, triplets, wts, ttl, hr)
        axes(ax); cla(ax);
        plot3(pts(:,1), pts(:,2), pts(:,3), 'o', ...
              'LineStyle','none','MarkerSize',8, ...
              'MarkerEdgeColor','k','MarkerFaceColor',[0.25 0.25 0.25]); hold on;

        cols = lines(size(triplets,1));
        for i = 1:size(triplets,1)
            idx = triplets(i,:); c = cols(i,:);
            plot3(pts(idx,1), pts(idx,2), pts(idx,3), '-', 'LineWidth', 2, 'Color', c);
            plot3([pts(idx(1),1), pts(idx(3),1)], ...
                  [pts(idx(1),2), pts(idx(3),2)], ...
                  [pts(idx(1),3), pts(idx(3),3)], '-', 'LineWidth', 2, 'Color', c);

            [~, posStrong] = min(vecnorm(pts(idx, :)', 2));
            sIdx = idx(posStrong);
            plot3(pts(sIdx,1), pts(sIdx,2), pts(sIdx,3), 'o', ...
                  'MarkerSize', 10, 'LineWidth', 2, ...
                  'MarkerEdgeColor','g','MarkerFaceColor','g');
        end

        labelWeights(pts, wts);
        plot3(0, 0, 0, 'p', 'MarkerSize', 12, 'MarkerEdgeColor','r','MarkerFaceColor','r');
        text(0, 0, 0, ' BS (0,0,0)', 'FontWeight','bold');

        xlim([-hr hr]); ylim([-hr hr]); zlim([-hr hr]);
        axis equal; grid on; view(45,25);
        set(ax,'Projection','orthographic','Layer','top');
        xlabel('X'); ylabel('Y'); zlabel('Z'); title(ttl);
    end

    function labelWeights(pts, wts)
        for ii = 1:size(pts,1)
            text(pts(ii,1), pts(ii,2), pts(ii,3), ...
                sprintf('%.2f', wts(ii)), ...
                'FontSize', 10, 'FontWeight', 'bold', ...
                'Color', 'w', 'EdgeColor', 'k', ...
                'BackgroundColor', 'k', 'Margin', 1, ...
                'HorizontalAlignment', 'left');
        end
    end
end
