%% Classification example (probit / logit)
%
% Description:
%   This code replicates the results in the appendix of the IHGP paper.
%   The code uses the GPML toolbox version 4.2 for comparing to the full EP
%   solution (http://www.gaussianprocess.org/gpml/code/matlab/doc/). The
%   toolbox codes are also leveraged in the moment calculations in ADF.
%
% Author:
%   2018 - Arno Solin
%

%% Dependencies

  % Clear all and close old plots
  clear, close all

  % The IHGP / state space codes
  addpath ../
  
  % GPML toolbox: Add the toolbox to your Matlab path and run its
  % startup script. The toolbox is used for the comparison to full EP and 
  % in the moment matching.
  % run('[path-to-gpml]/gpml-matlab-v4.2-2018-06-11/startup.m')

  
%% Data and model

  % Choose 'logit' or 'probit'
  mode = 'logit';
  
  % *** Simulate data ***
  rng(0,'twister')
  sigma2 = .01;
  x = linspace(0,6,100);
  y = sinc(6-x) + sqrt(sigma2)*randn(size(x));
  y = sign(y);
  xt = x;
    
  % Likelihood
  if strcmpi(mode,'logit')
    likfunc = @likLogistic;
  elseif strcmpi(mode,'probit')
    likfunc = @likErf;
  else
    error('Either logit or porbit!');  
  end
  
  
%% Full EP

  % We need specify the mean, covariance and likelihood functions
  meanfunc = @meanConst;
  covfunc = {'covMaterniso',3};
  inffunc = @infEP;

  % Set up priors
  prior.mean = {{@priorDelta}};     % The empirical mean is fixed

  % Finally we initialize the hyperparameter struct
  hyp = struct('mean', 0, 'cov', log([1 1]), 'lik', []);

  % Optimize hyperparameters
  hyp2 = minimize(hyp, @gp, -100, {@infPrior,inffunc,prior}, meanfunc, covfunc, likfunc, x(:), y(:));

  % To make predictions using these hyperparameters
  [mu_full,s2_full,fmu_full,fs2_full] = gp(hyp2, inffunc, meanfunc, covfunc, likfunc, x(:), y(:), xt(:));

  

%%  
  
  % Moments
  mom = @(mu,s2,k) feval(likfunc,[],y(k),mu,s2,'infEP');
  
  % State space model
  ss = @(x,p) cf_matern32_to_ss(p(1),p(2));
    
  % Hyperparameters initial
  w = [1 1];
  
  % Optimization options
  opts = optimset('GradObj','off','display','iter','DerivativeCheck','on');
  
  % Optimize hyperparameters w.r.t. log marginal likelihood
  [w2,ll] = fminunc(@(w) gf_adf(w,x,y,ss,mom), ...
      w,opts);
  
  % Hyperparameters
  %w2 = [log(exp(hyp2.cov(2))^2) hyp2.cov(1)];  
  
  % Optimize hyperparameters w.r.t. log marginal likelihood
  [w3,ll] = fminunc(@(w) ihgp_adf(w,x,y,ss,mom), ...
      w,opts);

%% Run ADF filtering/smoothing

  % ADF filtering
  tic
  [Eft1,Varft1,Covft1,lb1,ub1,out] = gf_adf(w2,x,y,ss,mom,x);
  toc
  
  % Infinite-horizon ADF
  tic
  [Eft2,Varft2,Covft2,lb2,ub2,out2] = ihgp_adf(w3,x,nan*y,ss,mom,x);
  toc

  
%% Visualize  

  % Custom colors
  color(1,:) = [ 68 114 181]/255; % blue
  color(2,:) = [ 42 171  97]/255; % green
  color(3,:) = [211  67  78]/255; % red
  color(4,:) = [133 112 183]/255; % violet
  color(5,:) = [207 185 105]/255; % okra
  color(6,:) = [ 70 183 208]/255; % teal

  % Lighter versions
  lcolor(1,:) = [176 196 226]/255; % blue
  lcolor(2,:) = [139 227 177]/255; % green
  lcolor(3,:) = [240 191 195]/255; % red
  lcolor(4,:) = [220 214 234]/255; % violet
  lcolor(5,:) = [244 239 221]/255; % okra
  lcolor(6,:) = [192 230 239]/255; % teal
  lcolor(7,:) = [7 7 7]/8;         % gray
  
  % Transofrmation
  if strcmpi(mode,'logit')
    linkf = @(t) 1./(1+exp(-t));
  elseif strcmpi(mode,'probit')
    linkf = @(p) normcdf(p);
  else
    error('Either logit or porbit!');  
  end
  
  figure(1); clf; hold on
    h=fill([x'; flipdim(x',1)], [linkf(ub2); flipdim(linkf(lb2),1)], [7 7 7]/8);
    h1=plot(x,linkf(Eft2),'-k');
    h3=plot(x,linkf(Eft1),'-', ...
            x,linkf(ub1),'--', ...
            x,linkf(lb1),'--','color',color(3,:));
    h2=plot(x',linkf(fmu_full),'-', ...
            x',linkf(fmu_full+1.96*sqrt(fs2_full)),'--', ...
            x',linkf(fmu_full-1.96*sqrt(fs2_full)),'--','color',color(1,:));        
    plot(x,(y+1)/2,'o','Color',color(2,:),'MarkerFaceColor',color(2,:))
    set(h,'EdgeColor',[7 7 7]/8)
    axis tight
    box on
    set(gca,'Layer','Top')
    legend([h2(3) h3(2) h(1)],'Full (EP)','State space (ADF)','IHGP (ADF)','Location','NorthOutside')
    ylim([-0.1 1.15])
    xlabel('Input, $t$'), ylabel('Output, $y$')
  