
clc;
clear;
close all;
rng default;

%% TRANSMISSION SHAFT OPTIMISATION
% x(1) = Outer diameter do (m)
% x(2) = Diameter ratio q = di/do
% Inner diameter di = q*do

T = 500;
M = 300;
L = 1;
G = 80e9;
rho = 7850;
S = 120e6;
thetaMax = pi/180;

% Initial design
x0 = [0.08 0.25];

% Bounds
lb = [0.03 0];
ub = [0.15 0.8];

% Objective function
obj = @(x) shaftObjective(x,rho,L);

% Nonlinear constraints
con = @(x) shaftConstraints(x,T,M,L,G,S,thetaMax);

% Penalty objective for PSO and SA
penalty = @(x) shaftPenalty(x,T,M,L,G,S,thetaMax);

names = {'fmincon'; 'GA'; 'Particle Swarm'; ...
    'Pattern Search'; 'Surrogate Optimisation'; ...
    'Simulated Annealing'};

X = nan(6,2);
Mass = nan(6,1);
Time = nan(6,1);
ExitFlag = nan(6,1);

%% 1. FMINCON
fprintf('\n1. Running FMINCON...\n');

opts = optimoptions('fmincon','Algorithm','sqp', ...
    'Display','off','MaxIterations',500);

tic;
[xx,ff,ee] = fmincon(obj,x0,[],[],[],[], ...
    lb,ub,con,opts);
Time(1) = toc;

X(1,:) = xx;
Mass(1) = ff;
ExitFlag(1) = ee;

%% 2. GENETIC ALGORITHM
fprintf('\n2. Running GA...\n');

opts = optimoptions('ga','Display','off', ...
    'PopulationSize',50,'MaxGenerations',100);

tic;
[xx,ff,ee] = ga(obj,2,[],[],[],[], ...
    lb,ub,con,opts);
Time(2) = toc;

X(2,:) = xx;
Mass(2) = ff;
ExitFlag(2) = ee;

%% 3. PARTICLE SWARM
fprintf('\n3. Running PARTICLESWARM...\n');

opts = optimoptions('particleswarm','Display','off', ...
    'SwarmSize',60,'MaxIterations',150);

tic;
[xx,~,ee] = particleswarm(penalty,2,lb,ub,opts);
Time(3) = toc;

X(3,:) = xx;
Mass(3) = obj(xx);
ExitFlag(3) = ee;

%% 4. PATTERN SEARCH
fprintf('\n4. Running PATTERNSEARCH...\n');

opts = optimoptions('patternsearch','Display','off', ...
    'MaxIterations',200);

tic;
[xx,ff,ee] = patternsearch(obj,x0,[],[],[],[], ...
    lb,ub,con,opts);
Time(4) = toc;

X(4,:) = xx;
Mass(4) = ff;
ExitFlag(4) = ee;

%% 5. SURROGATE OPTIMISATION
fprintf('\n5. Running SURROGATEOPT...\n');

opts = optimoptions('surrogateopt','Display','off', ...
    'MaxFunctionEvaluations',200);

tic;
[xx,~,ee] = surrogateopt( ...
    @(x)shaftSurrogate(x,T,M,L,G,S,thetaMax), ...
    lb,ub,[],opts);
Time(5) = toc;

X(5,:) = xx;
Mass(5) = obj(xx);
ExitFlag(5) = ee;

%% 6. SIMULATED ANNEALING
fprintf('\n6. Running SIMULANNEALBND...\n');

opts = optimoptions('simulannealbnd','Display','off', ...
    'MaxIterations',1000);

tic;
[xx,~,ee] = simulannealbnd(penalty,x0,lb,ub,opts);
Time(6) = toc;

X(6,:) = xx;
Mass(6) = obj(xx);
ExitFlag(6) = ee;

%% CALCULATE ACTUAL DESIGN DIMENSIONS
OuterDiameter = X(:,1);
InnerDiameter = X(:,1).*X(:,2);

%% CHECK FEASIBILITY AND ENGINEERING PERFORMANCE
Feasible = false(6,1);
VonMisesMPa = nan(6,1);
TwistDegrees = nan(6,1);

for k = 1:6

    if all(isfinite(X(k,:)))

        [c,~] = con(X(k,:));

        Feasible(k) = all(c <= 1e-6);

        do = OuterDiameter(k);
        di = InnerDiameter(k);

        D = do^4-di^4;

        sigma = 32*M*do/(pi*D);
        tau = 16*T*do/(pi*D);

        vm = sqrt(sigma^2+3*tau^2);

        theta = 32*T*L/(pi*G*D);

        VonMisesMPa(k) = vm/1e6;
        TwistDegrees(k) = theta*180/pi;

    end

end

%% RESULTS TABLE
Results = table(names,OuterDiameter,InnerDiameter, ...
    Mass,VonMisesMPa,TwistDegrees,Time,ExitFlag,Feasible, ...
    'VariableNames',{'Algorithm','OuterDiameter_m', ...
    'InnerDiameter_m','Mass_kg','VonMises_MPa', ...
    'Twist_Degrees','Time_s','ExitFlag','Feasible'});

disp('TRANSMISSION SHAFT OPTIMISATION RESULTS');
disp(Results);

writetable(Results,'TransmissionShaft_Results.xlsx');

%% EXECUTION TIME GRAPH
figure;
bar(Time);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Execution Time (seconds)');
title('Transmission Shaft - Execution Time Comparison');
grid on;

%% MASS COMPARISON GRAPH
figure;
bar(Mass);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Shaft Mass (kg)');
title('Transmission Shaft - Mass Comparison');
grid on;

%% LOCAL FUNCTIONS

function f = shaftObjective(x,rho,L)

do = x(1);
q = x(2);
di = q*do;

f = rho*pi/4*(do^2-di^2)*L;

end

function [c,ceq] = shaftConstraints(x,T,M,L,G,S,thetaMax)

do = x(1);
q = x(2);
di = q*do;

D = do^4-di^4;

sigma = 32*M*do/(pi*D);
tau = 16*T*do/(pi*D);

vonMises = sqrt(sigma^2+3*tau^2);

theta = 32*T*L/(pi*G*D);

c = [vonMises/S-1;
     theta/thetaMax-1];

ceq = [];

end

function f = shaftPenalty(x,T,M,L,G,S,thetaMax)

[c,~] = shaftConstraints(x,T,M,L,G,S,thetaMax);

f = shaftObjective(x,7850,L) ...
    + 1e4*sum(max(c,0).^2);

end

function out = shaftSurrogate(x,T,M,L,G,S,thetaMax)

[c,~] = shaftConstraints(x,T,M,L,G,S,thetaMax);

out.Fval = shaftObjective(x,7850,L);
out.Ineq = c(:)';

end
