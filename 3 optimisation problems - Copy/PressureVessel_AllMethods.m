
clc;
clear;
close all;
rng default;

%% PRESSURE VESSEL OPTIMISATION
% x(1) = Internal radius R (m)
% x(2) = Cylindrical length L (m)
% x(3) = Shell thickness ts (m)
% x(4) = Head thickness th (m)

P = 1e6;
S = 120e6;

x0 = [0.5 1 0.01 0.01];

lb = [0.3 0.2 0.005 0.005];
ub = [1.5 3 0.05 0.05];

obj = @(x) vesselObjective(x);
con = @(x) vesselConstraints(x,P,S);
penalty = @(x) vesselPenalty(x,P,S);

names = {'fmincon'; 'GA'; 'Particle Swarm'; ...
    'Pattern Search'; 'Surrogate Optimisation'; ...
    'Simulated Annealing'};

X = nan(6,4);
F = nan(6,1);
T = nan(6,1);
EXIT = nan(6,1);

%% 1. FMINCON
fprintf('\n1. Running FMINCON...\n');
opts = optimoptions('fmincon','Algorithm','sqp', ...
    'Display','off','MaxIterations',500);

tic;
[X(1,:),F(1),EXIT(1)] = fmincon(obj,x0,[],[],[],[], ...
    lb,ub,con,opts);
T(1) = toc;

%% 2. GENETIC ALGORITHM
fprintf('\n2. Running GA...\n');
opts = optimoptions('ga','Display','off', ...
    'PopulationSize',50,'MaxGenerations',100);

tic;
[X(2,:),F(2),EXIT(2)] = ga(obj,4,[],[],[],[], ...
    lb,ub,con,opts);
T(2) = toc;

%% 3. PARTICLE SWARM
fprintf('\n3. Running PARTICLESWARM...\n');
opts = optimoptions('particleswarm','Display','off', ...
    'SwarmSize',60,'MaxIterations',150);

tic;
[X(3,:),~,EXIT(3)] = particleswarm(penalty,4,lb,ub,opts);
T(3) = toc;
F(3) = obj(X(3,:));

%% 4. PATTERN SEARCH
fprintf('\n4. Running PATTERNSEARCH...\n');
opts = optimoptions('patternsearch','Display','off', ...
    'MaxIterations',200);

tic;
[X(4,:),F(4),EXIT(4)] = patternsearch(obj,x0,[],[],[],[], ...
    lb,ub,con,opts);
T(4) = toc;

%% 5. SURROGATE OPTIMISATION
fprintf('\n5. Running SURROGATEOPT...\n');
opts = optimoptions('surrogateopt', ...
    'Display','off','MaxFunctionEvaluations',200);

tic;
[X(5,:),~,EXIT(5)] = surrogateopt( ...
    @(x)vesselSurrogate(x,P,S),lb,ub,[],opts);
T(5) = toc;
F(5) = obj(X(5,:));

%% 6. SIMULATED ANNEALING
fprintf('\n6. Running SIMULANNEALBND...\n');
opts = optimoptions('simulannealbnd','Display','off', ...
    'MaxIterations',1000);

tic;
[X(6,:),~,EXIT(6)] = simulannealbnd( ...
    penalty,x0,lb,ub,opts);
T(6) = toc;
F(6) = obj(X(6,:));

%% RESULTS
Feasible = false(6,1);

for k = 1:6
    if all(isfinite(X(k,:)))
        [c,~] = con(X(k,:));
        Feasible(k) = all(c <= 1e-6);
    end
end

Results = table(names,X(:,1),X(:,2),X(:,3), ...
    X(:,4),F,T,EXIT,Feasible, ...
    'VariableNames',{'Algorithm','Radius','Length', ...
    'ShellThickness','HeadThickness','MaterialVolume', ...
    'TimeSeconds','ExitFlag','Feasible'});

disp('PRESSURE VESSEL OPTIMISATION RESULTS');
disp(Results);

writetable(Results,'PressureVessel_Results.xlsx');

figure;
bar(T);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Execution Time (seconds)');
title('Pressure Vessel - Algorithm Time Comparison');
grid on;

figure;
bar(F);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Material Volume (m^3)');
title('Pressure Vessel - Objective Comparison');
grid on;

%% LOCAL FUNCTIONS

function f = vesselObjective(x)

R = x(1);
L = x(2);
ts = x(3);
th = x(4);

f = pi*((R+ts)^2-R^2)*L ...
    + (4*pi/3)*((R+th)^3-R^3);

end

function [c,ceq] = vesselConstraints(x,P,S)

R = x(1);
L = x(2);
ts = x(3);
th = x(4);

V = pi*R^2*L + 4*pi*R^3/3;

c = [1-V;
     P*R/(ts*S)-1;
     P*R/(2*ts*S)-1;
     P*R/(2*th*S)-1];

ceq = [];

end

function f = vesselPenalty(x,P,S)

[c,~] = vesselConstraints(x,P,S);

f = vesselObjective(x) + ...
    1e6*sum(max(c,0).^2);

end

function out = vesselSurrogate(x,P,S)

out.Fval = vesselObjective(x);

[c,~] = vesselConstraints(x,P,S);

out.Ineq = c(:)';

end
