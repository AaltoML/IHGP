%% Coal mining example (log-Gaussian Cox process)
%
% Description:
%   This code replicates the results in Fig. 4 in the IHGP paper. This 
%   code uses the GPML toolbox version 4.2 for comparing to the full EP
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

  
%% Coal mining accidents example (full EP using GPML)

  % Load data
  x = load('coal.txt');

  % Discretization
  gridn = 200;

  % Discretize the data
  xx = linspace(min(x),max(x),gridn)';
  yy = hist(x,xx)';

  % Test points
  xt = xx;
  
  % We need specify the mean, covariance and likelihood functions
  meanfunc = @meanConst;
  covfunc = {'covMaterniso',5};
  likfunc = {@likPoisson,'exp'};    % Poisson likelihood
  inffunc = @infEP;

  % Set up priors
  prior.mean = {{@priorDelta}};     % The empirical mean is fixed

  % Finally we initialize the hyperparameter struct
  hyp = struct('mean', log(numel(x)/gridn), 'cov', log([1 .1]), 'lik', []);

  % Optimize hyperparameters
  hyp2 = minimize(hyp, @gp, -100, {@infPrior,inffunc,prior}, meanfunc, {@apxState,covfunc}, likfunc, xx, yy);

  % To make predictions using these hyperparameters
  [mu,s2,fmu,fs2,~,post] = gp(hyp2, inffunc, meanfunc, covfunc, likfunc, xx, yy, xt);

  
  % Compute mean and quantiles
  A = max(xx)-min(xx);
  scale = gridn/A;
  lm_gpml = exp(fmu+fs2/2)*scale; % == mu * scale
  lq5_gpml = exp(fmu-sqrt(fs2)*1.645)*scale;
  lq95_gpml = exp(fmu+sqrt(fs2)*1.645)*scale;

  figure(1); clf; hold on
    fill([xt; flipdim(xt,1)], [lq5_gpml; flipdim(lq95_gpml,1)], [7 7 7]/8)
    plot(xt, lm_gpml); 
    plot(x, 0*x, '+k')
    axis tight
    ylabel('Accident intensity')
    xlabel('Time (years)')

  Eft0 = fmu;
  ub0 = Eft0+1.96*sqrt(fs2);
  lb0 = Eft0-1.96*sqrt(fs2);
  
    
%% ADF

  % Set up moment calculations for ADF
  meanval = log(numel(x)/numel(xx));
  mom = @(mu,s2,k) feval(likfunc{:},[],yy(k),mu+meanval,s2,'infEP');

  % State space model
  ss = @(x,p) cf_matern52_to_ss(p(1),p(2));
  
  % Hyperparameters initial
  w = [log(exp(hyp.cov(2))^2) hyp.cov(1)];
  
  % Optimization options
  opts = optimset('GradObj','off','display','iter');
  
  % Optimize hyperparameters w.r.t. log marginal likelihood
  [w2,ll] = fminunc(@(w) gf_adf(w,xx,yy,ss,mom), ...
      w,opts);
  
  % Optimize hyperparameters w.r.t. log marginal likelihood
  [w3,ll] = fminunc(@(w) ihgp_adf(w,xx,yy,ss,mom), ...
      w,opts);
    
  

%% Run ADF filtering/smoothing for state space and IHGP

  % ADF filtering
  tic
  [Eft1,Varft1,Covft1,lb1,ub1,out] = gf_adf(w2,xx,yy,ss,mom,xx);
  toc
  
  % Infinite-horizon ADF
  tic
  [Eft2,Varft2,Covft2,lb2,ub2,out2] = ihgp_adf(w3,xx,yy,ss,mom,xx);
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
  
  % Compute mean and quantiles
  lm_kf = exp(Eft1+meanval+Varft1/2)*scale; % == mu * scale
  lq5_kf = exp(Eft1+meanval-sqrt(Varft1)*1.645)*scale;
  lq95_kf = exp(Eft1+meanval+sqrt(Varft1)*1.645)*scale;

  % Compute mean and quantiles
  lm_inf = exp(Eft2+meanval+Varft2/2)*scale; % == mu * scale
  lq5_inf = exp(Eft2+meanval-sqrt(Varft2)*1.645)*scale;
  lq95_inf = exp(Eft2+meanval+sqrt(Varft2)*1.645)*scale;

  figure(1); clf; hold on
    h1 = plot(xt, lm_gpml,'-k');   
    h = fill([xt; flip(xt,1)], [lq5_gpml; flip(lq95_gpml,1)], lcolor(5,:));
    plot(xt, lm_gpml,'-k'); 
    h2 = plot(xx,lm_kf,'--','color',color(3,:));
    h3 = plot(xx,lq95_kf,'--r',xx,lq5_kf,'--','color',color(3,:));
    set(h,'EdgeColor',lcolor(5,:))
    axis tight
    lims = ylim;
    plot(x, 0*x-.15, '+k','MarkerSize',3)    
    ylim(lims)
    ylabel('Accident intensity, \lambda(t)')
    xlabel('Time (years)')
    box on
    set(gca,'Layer','Top')
    set(gca,'XTick',1860:20:1960)
    legend([h1 h h2],'Full (mean)','Full (90% quantiles)','State space')
    
  figure(2); clf; hold on
    h1 = plot(xt, lm_inf,'-k'); 
    h = fill([xt; flip(xt,1)], [lq5_inf; flip(lq95_inf,1)], [7 7 7]/8);
    plot(xt, lm_inf,'-k'); 
    h2=plot(xx,lm_kf,'--','color',color(3,:));
    plot(xx,lq95_kf,'--r',xx,lq5_kf,'--','color',color(3,:))
    set(h,'EdgeColor',[7 7 7]/8)
    axis tight
    lims = ylim;
    plot(x, 0*x-.15, '+k','MarkerSize',3)    
    ylim(lims)
    ylabel('Accident intensity, \lambda(t)')
    xlabel('Time (years)')
    box on    
    set(gca,'Layer','Top')
    set(gca,'XTick',1860:20:1960)
    legend([h1 h h2],'IHGP (mean)','IHGP (90% quantiles)','State space')
    
