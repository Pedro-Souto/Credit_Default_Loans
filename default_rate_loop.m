%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IST - Inteligent Systems - MEIC
% Pedro Souto (N� 186332) @2016/2017
%
% generate a fuzzy model for the default rate of credit card clients
%
% Abstract: This research aimed at the case of customers
%           default payments in Taiwan
%
% Yeh, I. C., & Lien, C. H. (2009).
% The comparisons of data mining techniques for the predictive 
% accuracy of probability of default of credit card clients.
% Expert Systems with Applications, 36(2), 2473-2480.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       define constants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic; clear;
controlo = zeros(1,22); % To optimize FM.Nu
controlo_vars = zeros(1,7); % To optimize performance metrics
Nu_inicial = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]; %Initial/Starting Point
x = 0;
while x < 50;
for i=1:22
clear FM
clc;
FM.c = 3;      % number of clusters
FM.Ts = 1;     % sample time [s] 
FM.seed = 0;   % seed
FM.ante = 2;   % type of antecedent:  1 - product-space MFS
               %                      2 - projected MFS
FM.Ny = 0;     % denominator order
% numerator orders
FM.Nu = Nu_inicial;
FM.Nu(1,i)=0;
controlo(i,:) = FM.Nu;
% transport delays (set to 1 for y(k+1) = f(u(k),....)) 
FM.Nd = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
skip = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               read data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('raw_data.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
control = [LIMIT_BAL EDUCATION MARRIAGE AGE PAY_0 PAY_2 PAY_3 PAY_4 PAY_5...
           PAY_6 BILL_AMT1 BILL_AMT2 BILL_AMT3 BILL_AMT4 BILL_AMT5 BILL_AMT6...
           PAY_AMT1 PAY_AMT2 PAY_AMT3 PAY_AMT4 PAY_AMT5 PAY_AMT6];  % The varibale Sex is already excluded from here because of its binary condition

system = defaultpaymentnextmonth;
n2 = floor(size(system,1)/3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       identification data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
control_auxiliar = control(1:skip:end-n2,:);
system_auxiliar = system(1:skip:end-n2,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class_1 = [control_auxiliar system_auxiliar];
class_1(~all(class_1(:,end),2),:) = [];  % check if row isn't a default client
class_0 = [control_auxiliar system_auxiliar];
class_0(all(class_0(:,end),2),:) = [];  % check if row is a default client
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generator of a unbiased and random Sample (50% of Default and 50% of not default)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sample = [class_1; datasample(class_0,size(class_1,1))];
sample_final = datasample(sample,size(sample,1));
control_sample = sample_final(:,1:end-1);
system_sample = sample_final(:,end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dat.u = control_sample;
Dat.y = system_sample;
clearvars -except control system Dat FM skip n2 controlo controlo_vars i Nu_inicial x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           validation data
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
EVAL = Evaluate(ye(1:size(ym,1)),ym);
controlo_vars(i,1:6)= EVAL(:,1:end-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Optimization for the Fuzzy Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
controlo_vars(:,7) = mean(controlo_vars(:,1:6),2);
stats = array2table(controlo_vars);
stats.Properties.VariableNames = {'Accuracy' 'Sensitivity' 'Specificity'...
                                  'Precision' 'Recall' 'f_measure' 'Average'};
clearvars -except i j result FM Dat ye ue ym control system stats controlo controlo_vars i Nu_inicial x
[~,I] = max(controlo_vars(:,7));
stats(I,:)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Nu_inicial == controlo(I,:);
    x
    Nu_inicial
    break;
else
    Nu_inicial = controlo(I,:);
end
x = x +1;
end
clearvars -except control system Dat FM stats ue ye ym
toc;