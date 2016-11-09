%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IST - Inteligent Systems - MEIC
% Pedro Souto (Nº 186332) @2016/2017
%
% generate a fuzzy model for the default rate of credit card clients
%
% Abstract: This research aimed at the case of customers
%           default payments in Taiwan and compares the 
%           predictive accuracy of probability of default
%           among six data mining methods.
%
% Yeh, I. C., & Lien, C. H. (2009).
% The comparisons of data mining techniques for the predictive 
% accuracy of probability of default of credit card clients.
% Expert Systems with Applications, 36(2), 2473-2480.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define constants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic; clear;
controlo = zeros(1,23);
for i=1:23
    EVAL_aux = zeros(1,7);
    EVAL = ones(1,7);
    j = 2;
    while EVAL_aux(1,1) < EVAL(1,1) && EVAL_aux(1,2) < EVAL(1,2) &&...
          EVAL_aux(1,3) < EVAL(1,3) && EVAL_aux(1,7) < EVAL(1,7) &&...
          EVAL_aux(1,5) < EVAL(1,5) && EVAL_aux(1,6) < EVAL(1,6)
EVAL_aux = EVAL;
if j == -1, break, end;
clear FM
clc;
FM.c = 3;      % number of clusters
FM.Ts = 1;     % sample time [s] 
FM.seed = 0;   % seed
FM.ante = 2;   % type of antecedent:  1 - product-space MFS
               %                      2 - projected MFS
FM.Ny = 0;     % denominator order
% numerator orders
FM.Nu = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
% transport delays (set to 1 for y(k+1) = f(u(k),....)) 
FM.Nd = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]; 
FM.Nd(1,i)=j; controlo(1,i) = j;
skip = 1;    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read data(Windows and MAC Version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load('C:\Users\pedroalexsouto\Documents\MATLAB\1st_Project\raw_data.mat');
load('/Users/Pedroalexsouto/Documents/MATLAB/1st_Project/raw_data.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
control = [LIMIT_BAL EDUCATION MARRIAGE AGE PAY_0 PAY_2 PAY_3 PAY_4 PAY_5...
           PAY_6 BILL_AMT1 BILL_AMT2 BILL_AMT3 BILL_AMT4 BILL_AMT5 BILL_AMT6...
           PAY_AMT1 PAY_AMT2 PAY_AMT3 PAY_AMT4 PAY_AMT5 PAY_AMT6];  

system = defaultpaymentnextmonth;
n2 = floor(size(system,1)/3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% identification data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dat.u = control(1:skip:end-n2,:);
Dat.y = system(1:skip:end-n2,:);
clearvars -except control system Dat FM skip n2 VAF i j EVAL EVAL_aux controlo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% validation data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ue = control(end-n2+1:skip:end,:);
ye = system(end-n2+1:skip:end,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% make fuzzy model by means of fuzzy clustering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [FM,~] = fmclust(Dat,FM);
    [FM]=fmtune(FM);
    [FM,Part] = fmclust2(Dat,FM);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simulate the fuzzy model for validation data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1); clf;
[ym,VAF,dof,yl,ylm] = fmsim(ue,ye,FM,[],[],2);
title('Process output (blue) and model output (magenta)');
ylabel('Output');

figure(2); clf
subplot(211); plot([ylm{1}]);
axis([0 inf -1 1]);
title('Individual local models');
xlabel('Time'); ylabel('Output');

subplot(212); plot(dof{1})
title('Degrees of fulfillment');
xlabel('Time'); ylabel('Membership grade');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Statistics for the Fuzzy Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
decision = 0.5;
ym(ym<decision)=0; ym(ym>=decision)=1;
EVAL = 100.*Evaluate(ye(1:end-j),ym);
stats = array2table(EVAL);
stats.Properties.VariableNames = {'Accuracy' 'Sensitivity' 'Specificity'...
                                  'Precision' 'Recall' 'f_measure' 'gmean'};
stats
clearvars -except i j result FM Dat ye ue ym control system stats EVAL EVAL_aux controlo
j = j - 1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
toc;