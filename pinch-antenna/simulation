clear; clc; close all;

K = 2; %num users (must be even)
M = 4; %num pinch
rng('shuffle');

eta = 3;
P = 10;
N0 = 1e-5;
alpha1 = 0.6;
alpha2 = 0.4;
sigma = 1;
L = 100;

%coupling stuff
G = 6.0*ones(1,M);
r0 = 40;
gamma = 1.5;

%antenna seq share param
phi_ant = 0.6; %something like 0.5 to 0.7

%penalty with more antennas
tau_sic = 0.04; %pen per extra active antenn (0.02 to 0.06)

BS = [0, 0];

%or equally spaced on a circle
R = 70; theta = linspace(0, 2*pi, M+1); theta(end) = [];
A = [R*cos(theta(:)), R*sin(theta(:))];

%helpers
rayleigh = @(s) s*sqrt(-2*log(max(1e-12,rand)));
dist = @(X,Y) sqrt(sum((X - Y).^2, 2));
coupling_fun = @(d) 1 ./ (1 + (d./r0).^gamma); %user antenna

%noma without penalty
noma_pair_rates_raw = @(hw, hs) deal( ...
    log2(1 + (alpha1*P*hw.^2) ./ (alpha2*P*hw.^2 + N0)), ... %weak
    log2(1 + (alpha2*P*hs.^2) ./ N0) ); %strong

%antenna line share
line_weights = @(a) compute_line_weights(a, phi_ant);

U = (rand(K,2)*2 - 1) * L;

%bs to user
dBU = dist(U, repmat(BS,K,1));
z = arrayfun(@(i) rayleigh(sigma), 1:K).';
h_direct = z ./ (max(1e-3, dBU).^(eta/2));    % Kx1

%user to antenna coupling matrix
C = zeros(K,M);
for m = 1:M
    C(:,m) = coupling_fun( dist(U, repmat(A(m,:),K,1)) );
end

%search for best antenna activation with sic penalty
if M <= 6
    %brute force
    allA = dec2bin(0:(2^M-1)) - '0';
    bestS = -inf; bestA = [];
    for r = 1:size(allA,1)
        a = allA(r,:);
        S = sumrate_for_activation(a, h_direct, C, G, noma_pair_rates_raw, line_weights, tau_sic);
        if S > bestS
            bestS = S; bestA = a;
        end
    end
else
    %otherwise greedy with on off
    a = zeros(1,M);
    bestS = sumrate_for_activation(a, h_direct, C, G, noma_pair_rates_raw, line_weights, tau_sic);
    improved = true;
    while improved
        improved = false;
        bestLocalS = bestS;
        bestLocalA = a;
        for m = 1:M
            a_try = a; a_try(m) = 1 - a_try(m);
            S_try = sumrate_for_activation(a_try, h_direct, C, G, noma_pair_rates_raw, line_weights, tau_sic);
            if S_try > bestLocalS
                bestLocalS = S_try;
                bestLocalA = a_try;
                improved = true;
            end
        end
        a = bestLocalA;
        bestS = bestLocalS;
    end
    bestA = a;
end

%best activation metrics
w_best = line_weights(bestA);
g_mul_best = 1 + (C * (G(:).*w_best(:)));
h_eff_best = h_direct .* g_mul_best;

%pairing under best activation
[~, idxS] = sort(abs(h_eff_best), 'ascend');
h_sorted  = h_eff_best(idxS);
hw = h_sorted(1:K/2);
hs = flipud(h_sorted(K/2+1:end));

%rates without pen
[Rw_raw, Rs_raw] = noma_pair_rates_raw(hw, hs);

%pen to strong user rate
M_active = sum(bestA);
overhead = max(0, 1 - tau_sic * max(0, M_active-1));

Rw = Rw_raw;% weak unaffected bc no sic
Rs = Rs_raw * overhead; %strong reduced by overhead
pair_sumrates = Rw + Rs;

%results
fprintf('K = %d users, M = %d antennas\n', K, M);
fprintf('Active antennas: %d\n', M_active);
fprintf('Best Activation: [%s]\n', sprintf('%d', bestA));
fprintf('SIC overhead factor (strong user): %.3f\n', overhead);
fprintf('Sequential line share weights (active only): [ %s]\n', sprintf('%.3f ', w_best(w_best>0)));
for p = 1:min(5, K/2)
    fprintf('Pair %d: Rw = %.3f, Rs = %.3f (raw %.3f), Sum = %.3f\n', ...
        p, Rw(p), Rs(p), Rs_raw(p), pair_sumrates(p));
end
fprintf('Total Sum Rate: %.3f b/s/Hz\n', sum(pair_sumrates));
fprintf('Mean Per Pair Sum Rate: %.3f b/s/Hz\n', mean(pair_sumrates));

%plot
figure('Color','w'); hold on; axis equal;
xlim([-L L]); ylim([-L L]);
xline(0,'k-','LineWidth',1); yline(0,'k-','LineWidth',1); box off;

%bs + antennas
plot(BS(1),BS(2),'ks','MarkerFaceColor','k','MarkerSize',9); text(BS(1)+2,BS(2),'BS');

for m = 1:M
    face = bestA(m)*[0 0 0] + [1 1 1]; % filled if active
    plot(A(m,1), A(m,2), '^', 'MarkerSize', 9, 'MarkerFaceColor', face, 'Color', 'k');
    if bestA(m)==1
        txt = sprintf('A%d (w=%.2f)', m, w_best(m));
    else
        txt = sprintf('A%d', m);
    end
    text(A(m,1)+2, A(m,2), txt);
end

%users / links
userColors = lines(K);
plot(U(:,1),U(:,2),'o','MarkerFaceColor',[0.2 0.6 1],'Color',[0 0.2 0.6]);
for i = 1:K
    text(U(i,1)+2, U(i,2), sprintf('U%d', i));
end
for i = 1:K
    plot([BS(1) U(i,1)], [BS(2) U(i,2)], ':', 'Color', userColors(i,:), 'LineWidth', 1.2);
end
for m = 1:M
    if bestA(m)==1
        for i = 1:K
            plot([A(m,1) U(i,1)], [A(m,2) U(i,2)], '-', 'Color', userColors(i,:), 'LineWidth', 1.2);
        end
    end
end

title(sprintf('Best Activation [%s] | %d Active | Total Sum Rate = %.3f', ...
      sprintf('%d',bestA), M_active, sum(pair_sumrates)));
xlabel('x (m)'); ylabel('y (m)'); grid on;

%more funcs
function S = sumrate_for_activation(a, hdir, Cmat, Gvec, rate_fn_raw, line_weights_fn, tau_sic)
    %sequential line share weights for current activation
    w = line_weights_fn(a);
    g_mul = 1 + (Cmat * (Gvec(:).*w(:)));
    h_eff = hdir .* g_mul;

    %pair weakest with strongest
    [~, idx] = sort(abs(h_eff), 'ascend');
    h_sorted = h_eff(idx);
    Kloc = numel(h_sorted);
    hw = h_sorted(1:Kloc/2);
    hs = flipud(h_sorted(Kloc/2+1:end));

    %raw noma rates
    [Rw_raw, Rs_raw] = rate_fn_raw(hw, hs);

    %sic complexity penalty depends on num active antennas
    M_active = sum(a);
    overhead = max(0, 1 - tau_sic * max(0, M_active-1));
    Rw = Rw_raw;
    Rs = Rs_raw * overhead;

    S = sum(Rw + Rs);
end

function w = compute_line_weights(a, phi)
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
