clear;
clc;
close all;

K = 2; %num users (must be even)
M_list = 0:20; %antennas
trials = 500; %increase for smoother curves
rng(1);

eta = 3;
P = 10;
N0 = 1e-5;
alpha1 = 0.6;
alpha2 = 0.4;
sigma = 1;
L = 100;

%coupling stuff
G_base = 6.0;
r0 = 40;
gamma = 1.5;

%antenna seq share param
phi_ant = 0.6; %something like 0.5 to 0.7

%complexity penalty settings
penalty_on_default = true;
penalty_scale = 1.0;
BRUTE_M_MAX = 6; %if over 6 ant then use greedy

%store
avg_sumrate_pen = zeros(size(M_list)); %pen
avg_sumrate_nopen = zeros(size(M_list)); %no pen
avg_Mactive_pen = zeros(size(M_list)); %avg num active actually used (pen)
avg_Mactive_nopen = zeros(size(M_list)); % avg num active used (no pen)

%main sweep
for mi = 1:numel(M_list)
    M = M_list(mi);

    sr_pen_acc   = 0;
    sr_nopen_acc = 0;
    Mact_pen_acc = 0;
    Mact_nopen_acc = 0;

    for t = 1:trials
        U = (rand(K,2)*2 - 1) * L;

        %bs, antenna array
        BS = [0, 0];
        if M > 0
            R = 70;
            theta = linspace(0, 2*pi, M+1); theta(end) = [];
            A = [R*cos(theta(:)), R*sin(theta(:))];
        else
            A = zeros(0,2); %no ant
        end

        %helpers
        dist = @(X,Y) sqrt(sum((X - Y).^2, 2));
        rayleigh = @(s) s*sqrt(-2*log(max(1e-12,rand)));
        coupling_fun = @(d) 1 ./ (1 + (d./r0).^gamma);

        %bs to user
        dBU = dist(U, repmat(BS,K,1));
        z   = arrayfun(@(~) rayleigh(sigma), 1:K).';
        h_direct = z ./ (max(1e-3, dBU).^(eta/2));

        %user antenna coupling
        C = zeros(K,M);
        for m = 1:M
            C(:,m) = coupling_fun( dist(U, repmat(A(m,:),K,1)) );
        end

        %antenna line share
        line_weights = @(a) compute_line_weights(a, phi_ant);

        %rates
        rate_fn_pen   = @(hw,hs,Mact) noma_pair_rates_with_complexity( ...
            hw, hs, Mact, P, N0, alpha1, alpha2, true,  penalty_scale);
        rate_fn_nopen = @(hw,hs,Mact) noma_pair_rates_with_complexity( ...
            hw, hs, Mact, P, N0, alpha1, alpha2, false, penalty_scale);

        %best activation considering pen
        [S_pen, bestA_pen] = best_activation(M, h_direct, C, G_base*ones(1,M), ...
            line_weights, rate_fn_pen, BRUTE_M_MAX);

        %best act without pen
        [S_nopen, bestA_nopen] = best_activation(M, h_direct, C, G_base*ones(1,M), ...
            line_weights, rate_fn_nopen, BRUTE_M_MAX);

        %sum
        sr_pen_acc = sr_pen_acc + S_pen;
        sr_nopen_acc = sr_nopen_acc + S_nopen;
        Mact_pen_acc = Mact_pen_acc + sum(bestA_pen);
        Mact_nopen_acc = Mact_nopen_acc + sum(bestA_nopen);
    end

    %avg
    avg_sumrate_pen(mi) = sr_pen_acc / trials;
    avg_sumrate_nopen(mi) = sr_nopen_acc / trials;
    avg_Mactive_pen(mi) = Mact_pen_acc / trials;
    avg_Mactive_nopen(mi) = Mact_nopen_acc / trials;
end

%find optimal
[best_val, best_idx] = max(avg_sumrate_pen);
best_M = M_list(best_idx);

fprintf('Optimal Number of Antennas: M = %d\n', best_M);
fprintf('Avg Sum Rate = %.3f b/s/Hz\n', best_val);
fprintf('Avg Activated Antennas @ M = %d: %.2f\n', best_M, avg_Mactive_pen(best_idx));

%plot
figure('Color','w'); hold on; box on; grid on;
plot(M_list, avg_sumrate_pen,   'o-','LineWidth',1.8,'DisplayName','Sum Rate (Penalty)');
plot(M_list, avg_sumrate_nopen, 's-','LineWidth',1.8,'DisplayName','Sum Rate (No Penalty)');
xline(best_M,'k--','DisplayName',sprintf('Optimal M=%d (Penalty)',best_M));
xlabel('Number of Available Antennas (M)'); ylabel('Average Total Sum Rate (b/s/Hz)');
title(sprintf('Sum Rate v Antennas | %d Users, %d Trials, Penalty Scale = %.2f', K, trials, penalty_scale));
legend('Location','best');

%plot # antennas optimizer actually uses
figure('Color','w'); hold on; box on; grid on;
plot(M_list, avg_Mactive_pen,   'o-','LineWidth',1.8,'DisplayName','Avg Active (penalty)');
plot(M_list, avg_Mactive_nopen, 's-','LineWidth',1.8,'DisplayName','Avg Active (no penalty)');
plot(M_list, M_list, ':', 'LineWidth',1.2,'DisplayName','All available');
xlabel('Number of Available Antennas (M)'); ylabel('Avg # Active Antennas Used');
title('Activation Behavior v Available Antennas');
legend('Location','northwest');

%more funcs
function [S_best, bestA] = best_activation(M, h_direct, C, G, line_weights_fn, rate_fn, BRUTE_M_MAX)
    %best activation patterns
    if M == 0
        %no ant
        S_best = sumrate_for_activation(zeros(1,0), h_direct, C, G, rate_fn, line_weights_fn);
        bestA = zeros(1,0);
        return;
    end

    %brute force
    if M <= BRUTE_M_MAX
        allA = dec2bin(0:(2^M-1)) - '0';
        S_best = -inf; bestA = zeros(1,M);
        for r = 1:size(allA,1)
            a = allA(r,:);
            S_try = sumrate_for_activation(a, h_direct, C, G, rate_fn, line_weights_fn);
            if S_try > S_best
                S_best = S_try; bestA = a;
            end
        end
    else
        %greedy flip
        a = zeros(1,M);
        S_best = sumrate_for_activation(a, h_direct, C, G, rate_fn, line_weights_fn);
        improved = true;
        while improved
            improved = false;
            bestLocalS = S_best; bestLocalA = a;
            for m = 1:M
                a_try = a; a_try(m) = 1 - a_try(m);
                S_try = sumrate_for_activation(a_try, h_direct, C, G, rate_fn, line_weights_fn);
                if S_try > bestLocalS
                    bestLocalS = S_try; bestLocalA = a_try; improved = true;
                end
            end
            a = bestLocalA; S_best = bestLocalS;
        end
        bestA = a;
    end
end

function S = sumrate_for_activation(a, hdir, Cmat, Gvec, rate_fn, line_weights_fn)
    %line share weights for curr activation
    w = line_weights_fn(a);
    g_mul = 1 + (Cmat * (Gvec(:).*w(:)));
    h_eff = hdir .* g_mul;

    %pair weak to strongest
    [~, idx] = sort(abs(h_eff), 'ascend');
    h_sorted = h_eff(idx);
    Kloc = numel(h_sorted);
    hw = h_sorted(1:Kloc/2);
    hs = flipud(h_sorted(Kloc/2+1:end));

    %num active ant
    Mact = sum(a);
    [Rw, Rs] = rate_fn(hw, hs, Mact);
    S = sum(Rw + Rs);
end

function w = compute_line_weights(a, phi)
    %earlier ant indices = priority on the line.
    Mloc = numel(a);
    w = zeros(1, Mloc);
    rem = 1.0;
    for m = 1:Mloc
        if a(m) == 1
            w(m) = phi * rem;
            rem = rem - w(m);
            if rem <= 1e-12
                rem = 0;
            end
        end
    end
end

%compute sic penalty
function [Rw, Rs] = noma_pair_rates_with_complexity(hw, hs, Mact, P, N0, a1, a2, penalty_on, penalty_scale)
    sinr_w = (a1*P.*(hw.^2)) ./ (a2*P.*(hw.^2) + N0);
    sinr_s = (a2*P.*(hs.^2)) ./ N0;

    if penalty_on
        denom = penalty_scale * log2(1 + max(0, Mact));
        if denom < 1, denom = 1; end
        sinr_s = sinr_s ./ denom;
    end

    Rw = log2(1 + sinr_w);
    Rs = log2(1 + sinr_s);
end
