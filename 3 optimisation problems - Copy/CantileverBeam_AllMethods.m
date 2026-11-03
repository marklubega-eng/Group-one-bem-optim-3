
clc;
clear;
close all;
rng default;

%% CANTILEVER BEAM OPTIMISATION
% x(1) = Width b (m)
% x(2) = Height h (m)

F = 1000;
L = 2;
E = 200e9;
rho = 7850;
S = 150e6;
deltaMax = 0.005;

% Initial design
x0 = [0.05 0.10];

% Lower and upper bounds
lb = [0.02 0.02];
ub = [0.15 0.20];

% Objective function
obj = @(x) rho*L*x(1)*x(2);

% Constraint function
con = @(x) beamConstraints(x,F,L,E,S,deltaMax);

% Penalty objective for PSO and SA
penalty = @(x) beamPenalty(x,F,L,E,S,deltaMax);

names = {'fmincon'; 'GA'; 'Particle Swarm'; ...
    'Pattern Search'; 'Surrogate Optimisation'; ...
    'Simulated Annealing'};

X = nan(6,2);
Mass = nan(6,1);
Time = nan(6,1);
ExitFlag = nan(6,1);

%% 1. FMINCON
fprintf('\n1. Running FMINCON...\n');

opts = optimoptions('fmincon', ...
    'Algorithm','sqp','Display','off', ...
    'MaxIterations',500);

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
    @(x)beamSurrogate(x,F,L,E,S,deltaMax), ...
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

%% CHECK FEASIBILITY
Feasible = false(6,1);
StressMPa = nan(6,1);
Deflection = nan(6,1);

for k = 1:6
    if all(isfinite(X(k,:)))
        b = X(k,1);
        h = X(k,2);

        StressMPa(k) = 6*F*L/(b*h^2)/1e6;
        Deflection(k) = 4*F*L^3/(E*b*h^3);

        [c,~] = con(X(k,:));
        Feasible(k) = all(c <= 1e-6);
    end
end

%% DISPLAY RESULTS
Results = table(names,X(:,1),X(:,2),Mass, ...
    StressMPa,Deflection,Time,ExitFlag,Feasible, ...
    'VariableNames',{'Algorithm','Width','Height', ...
    'Mass_kg','Stress_MPa','Deflection_m', ...
    'Time_s','ExitFlag','Feasible'});

disp('CANTILEVER BEAM OPTIMISATION RESULTS');
disp(Results);

writetable(Results,'CantileverBeam_Results.xlsx');

%% COMPARISON GRAPHS
figure;
bar(Time);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Execution Time (seconds)');
title('Cantilever Beam - Execution Time Comparison');
grid on;

figure;
bar(Mass);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Beam Mass (kg)');
title('Cantilever Beam - Mass Comparison');
grid on;

%% LOCAL FUNCTIONS

function [c,ceq] = beamConstraints(x,F,L,E,S,deltaMax)

b = x(1);
h = x(2);

stress = 6*F*L/(b*h^2);
deflection = 4*F*L^3/(E*b*h^3);

% Normalised inequality constraints
c = [stress/S - 1;
     deflection/deltaMax - 1];

ceq = [];

end

function f = beamPenalty(x,F,L,E,S,deltaMax)

[c,~] = beamConstraints(x,F,L,E,S,deltaMax);

f = 7850*L*x(1)*x(2) ...
    + 1e5*sum(max(c,0).^2);

end

function out = beamSurrogate(x,F,L,E,S,deltaMax)

b = x(1);
h = x(2);

stress = 6*F*L/(b*h^2);
deflection = 4*F*L^3/(E*b*h^3);

out.Fval = 7850*L*b*h;

out.Ineq = [stress/S-1, ...
            deflection/deltaMax-1];

end
