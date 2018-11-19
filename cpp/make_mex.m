%% MEX the C++ function for evaluating the marginal likelihood
%
% Run this file in Matlab for compiling the mex function.
  
    mex -v CFLAGS="\$CFLAGS -Wall" LDFLAGS="\$LDFLAGS -w" ...
      -I/usr/local/Cellar/eigen/3.3.4/include/eigen3 ...
      ihgpr_mex.cpp InfiniteHorizonGP.cpp
  
  
%% Smoke test
%{
  [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_matern32_to_ss(1, 1);
  
  R = 1;
  dR = rand(1,1,size(dF,3));
  dt = 1;
  y = rand(10,1);
  [e,eg]=gf_stat_solve_mex(y,dt,F,H,Pinf,R,dF,dPinf,dR);
%}