%% Toy example (sinc function)
%
% Description:
%   This code does GP regression using toy data from a sinc function and
%   produces the results in Fig. 1 in the IHGP paper.
%
% Author:
%   2018 - Arno Solin
%
%% Run full GP and infinite-horizon

  % *** Dependencies ***
  addpath ../


  % *** Color scheme ***

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

  
  % *** Simulate data ***
  
  rng(0,'twister')
  sigma2 = .1;
  x = linspace(0,6,100);
  y = sinc(6-x) + sqrt(sigma2)*randn(size(x));
  
  
  % *** Initial model parameters ***

  % Initial parameters (sigma2, magnSigma2, lengthScale)
  param = [.1 .1 .1];

  
  % *** Set up full model ***  
  
  % Covariance function (Matern, nu=3/2)
  k = @(r,p) p(1)*(1+sqrt(3)*abs(r)/p(2)).*exp(-sqrt(3)*abs(r)/p(2));
  
  % Derivatives of covariance function (Matern, nu=3/2)
  dk{1} = @(r,p) (1+sqrt(3)*abs(r)/p(2)).*exp(-sqrt(3)*abs(r)/p(2));
  dk{2} = @(r,p) p(1)*3*r.^2/p(2)^3.*exp(-sqrt(3)*abs(r)/p(2));
  
  
  % *** Optimize hyperparameters and predict: Full GP ***
    
  % Optimization options
  opts = optimset('GradObj','on','display','iter');
  
  % Optimize hyperparameters w.r.t. log marginal likelihood
  [w1,ll] = fminunc(@(w) gp_solve(w,x,y,k,dk), ...
      log(param),opts);
  
  % Solve
  [Eft1,Varft1,Covft1,lb1,ub1] = gp_solve(w1,x,y,k,x);
  
  
  % *** Set up equivalent state space model ***
  
  % State space model
  ss = @(x,p) cf_matern32_to_ss(p(1),p(2));
  
  
  % *** Optimize hyperparameters and predict: IHGP ***
  
  % Optimization options
  opts = optimset('GradObj','on','display','iter');
  
  % Optimize hyperparameters w.r.t. log marginal likelihood
  [w2,ll] = fminunc(@(w) ihgpr(w,x,y,ss), ...
      log(param),opts);
    
  % Solve Infinite-Horizon GP regression problem
  [Eft2,Varft2,Covft2,lb2,ub2,out] = ihgpr(w2,x,y,ss,x);
  
  
  % *** Visualize results
  
  figure(2); clf; hold on
    h2=plot(x,Eft2,'-k');
    h0 = fill([x'; flip(x',1)], [ub2; flip(lb2,1)], lcolor(end,:));
    set(h0,'EdgeColor',lcolor(end,:))
    plot(x,Eft2,'-k'); 
    h1=plot(x,Eft1,'--',x,lb1,'--',x,ub1,'--','color',color(3,:));
    plot(x,y,'+k','MarkerSize',3)
    ylim([-1 1.2])
    legend([h1(1) h0 h2(1)],'Full GP','95% quantiles','IHGP mean','Location','SE')
    xlabel('Input, t'), ylabel('Output, y')
    box on
    set(gca,'Layer','Top')
    

    