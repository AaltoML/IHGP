function [varargout] = gf_solve(w,x,y,ss,xt,filteronly)
% GF_SOLVE - Solve GP regression problem by filtering
%
% Syntax:
%   [...] = gf_solve(w,x,y,k,xt)
%
% In:
%   w     - Log-parameters (nu, sigma2, theta)
%   x     - Training inputs
%   y     - Training outputs
%   ss    - State space model function handle, [F,L,Qc,...] = @(x,theta) 
%   xt    - Test inputs (default: empty)
%   filteronly - Run only filter (default: false)
%
% Out (if xt is empty or not supplied):
%
%   e     - Negative log marginal likelihood
%   eg    - ... and its gradient
%
% Out (if xt is not empty):
%
%   Eft   - Predicted mean
%   Varft - Predicted marginal variance
%   Covft - Predicted joint covariance matrix
%   lb    - 95% confidence lower bound
%   ub    - 95% confidence upper bound
%
% Description:
%   Consider the following GP regression problem:
%
%       f ~ GP(0,k(x,x')),
%     y_i = f(x_i),  i=1,2,...,n,
%
%   where k(x,x') = k_theta(x,x') + sigma2*delta(x,x').
%   The state space model is specified by the function handle 'ss' such
%   that it returns the state space model matrices
%
%     [F,L,Qc,H,Pinf,dF,dQc,dPinf] = ss(x,theta),
%
%   where theta holds the hyperparameters. See the paper for details.
%   This code is assuming a stationary covariance function even though the
%   methodology per se does not require it.
%
%   NOTE: This code is proof-of-concept, not optimized for speed.
%
% References:
%
%   [1] Simo Sarkka, Arno Solin, Jouni Hartikainen (2013).
%       Spatiotemporal learning via infinite-dimensional Bayesian
%       filtering and smoothing. IEEE Signal Processing Magazine,
%       30(4):51-61.
%
% Copyright:
%   2014-2015   Arno Solin
%
%  This software is distributed under the GNU General Public
%  License (version 3 or later); please refer to the file
%  License.txt, included with the software, for details.

%% Check defaults

  % Is there test data
  if nargin < 5, xt = []; end
  
  % Is filteronly set
  if nargin < 6 || isempty(filteronly), filteronly = false; end
  
  % Jitter sigma2
  jitterSigma2 = 1e-9;  
  
  
%% Figure out the correct way of dealing with the data

  % Combine observations and test points
  xall = [x(:); xt(:)];
  yall = [y(:); nan(numel(xt),1)];
    
  % Make sure the points are unique and in ascending order
  [xall,sort_ind,return_ind] = unique(xall,'first');
  yall = yall(sort_ind);
  
  % Only return test indices
  return_ind = return_ind(end-numel(xt)+1:end);
  
  
%% Set up model

  % Log transformed parameters
  param = exp(w);

  % Extract values
  n      = numel(x);
  sigma2 = param(1);
  
  % Form the state space model
  [F,L,Qc,H,Pinf,dF,dQc,dPinf] = ss(x,param(2:end));
  
  Hs = H;
  R = sigma2;
  
  % Concatenate derivatives
  dF    = cat(3,zeros(size(F)),dF);
  dQc   = cat(3,zeros(size(Qc)),dQc);
  dPinf = cat(3,zeros(size(Pinf)),dPinf);
  dR    = zeros(1,1,numel(param)); dR(1) = 1;  

  
%% Prediction of test inputs (filtering and smoothing)

  % Check that we are predicting
  if ~isempty(xt)

    % Set initial state
    m = zeros(size(F,1),1);
    P = Pinf;
    
    % Allocate space for results
    MS = zeros(size(m,1),size(yall,1));
    PS = zeros(size(m,1),size(m,1),size(yall,1));
    MP = MS;
    PP = PS;
    A = zeros(size(F,1),size(F,2),size(yall,1));
    Q = zeros(size(P,1),size(P,2),size(yall,1));
    KS = zeros(size(P,1),1,size(yall,1));
    
    % Initial dt
    dt = inf;
    
    % ### Forward filter
    
    % The filter recursion
    for k=1:numel(yall)
        
        % Solve A using the method by Davison
        if (k>1)
            
            % Discrete-time solution (only for stable systems)
            dt_old = dt;
            dt = xall(k)-xall(k-1);
            
            % Should we calculate a new discretization?
            if abs(dt-dt_old) < 1e-12
                A(:,:,k) = A(:,:,k-1);
                Q(:,:,k) = Q(:,:,k-1);
            else
                A(:,:,k) = expm(F*dt);
                Q(:,:,k) = Pinf - A(:,:,k)*Pinf*A(:,:,k)';
            end
            
            % Prediction step
            m = A(:,:,k) * m;
            P = A(:,:,k) * P * A(:,:,k)' + Q(:,:,k);
            
        end
        
        % Store predicted mean
        MP(:,k) = m;
        PP(:,:,k) = P;
        
        % Update step
        if ~isnan(yall(k))
            
            S = H*P*H' + R;
            K = P*H'/S;
            v = yall(k,:)'-H*m;
            m = m + K*v;
            P = P - K*H*P;
            KS(:,:,k) = K;
            
        end
        
        % Store estimate
        MS(:,k)   = m;
        PS(:,:,k) = P;
        
    end
       
    % ### Backward smoother

    GS = [];
    
    % Should we run the smoother?
    if ~filteronly
      
    % Allocate space for storing the smoother gain matrix
    GS = zeros(size(F,1),size(F,2),size(yall,1));

    % Rauch-Tung-Striebel smoother
    for k=size(MS,2)-1:-1:1
      
      % Smoothing step (using Cholesky for stability)
      PSk = PS(:,:,k);
      
      % Pseudo-prediction
      PSkp = A(:,:,k+1)*PSk*A(:,:,k+1)'+Q(:,:,k+1);
      
      % Solve the Cholesky factorization
      [L,notposdef] = chol(PSkp,'lower');
      
      % Numerical problems in Cholesky, retry with jitter
      if notposdef>0
          jitter = sqrt(jitterSigma2)*diag(rand(size(A,1),1));
          L = chol(PSkp+jitter,'lower');
      end
      
      % Continue smoothing step
      G = PSk*A(:,:,k+1)'/L'/L;
      
      % Do update
      m = MS(:,k) + G*(m-A(:,:,k+1)*MS(:,k));
      P = PSk + G*(P-PSkp)*G';
      
      % Store estimate
      MS(:,k)   = m;
      PS(:,:,k) = P;
      GS(:,:,k) = G;
      
    end
    
    end
    
    if nargout>3
      out.K = KS;
      out.G = GS;
    end
    Covft = [];
    
    % These indices shall remain to be returned
    MS = MS(:,return_ind);
    PS = PS(:,:,return_ind);
    
    % Return mean
    Eft = Hs*MS;
    
    % Return variance
    if nargout > 1
        Varft = zeros(size(Hs,1),size(Hs,1),size(MS,2));
        for k=1:size(MS,2)
            Varft(:,:,k)  = Hs*PS(:,:,k)*Hs';
        end
    end
    
    % Return values
    varargout = {Eft(:),Varft(:),out};
 
    % Also return joint covariance and upper/lower 95% bounds
    if nargout > 3

        % The bounds
        lb = Eft(:) - 1.96*sqrt(Varft(:));
        ub = Eft(:) + 1.96*sqrt(Varft(:));
        varargout = {Eft(:),Varft(:),Covft,lb(:),ub(:),out};
        
    end

  end
  
  
%% Evaluate negative log marginal likelihood and its gradient

  if isempty(xt)
  
    % Size of inputs
    d = size(F,1);
    nparam = numel(param);
    steps = numel(yall);
            
    % Allocate space for results
    edata = 0;
    gdata = zeros(1,nparam);
    
    % Set up
    Z  = zeros(d);
    m  = zeros(d,1);
    P  = Pinf;
    dm = zeros(d,nparam);
    dP = dPinf;
    dt = -inf;
    
    % Allocate space for expm results
    AA = zeros(2*d,2*d,nparam);
    
    % Loop over all observations
    for k=1:steps
        
        % The previous time step
        dt_old = dt;
        
        % The time discretization step length
        if (k>1)
            dt = xall(k)-xall(k-1);
        else
            dt = 0;
        end
        
        % Loop through all parameters (Kalman filter prediction step)
        for j=1:nparam
            
            % Should we recalculate the matrix exponential?
            if abs(dt-dt_old) > 1e-9
                
                % The first matrix for the matrix factor decomposition
                FF = [ F        Z;
                      dF(:,:,j) F];
                
                % Solve the matrix exponential
                AA(:,:,j) = expm(FF*dt);
                
            end
            
            % Solve the differential equation
            foo     = AA(:,:,j)*[m; dm(:,j)];
            mm      = foo(1:d,:);
            dm(:,j) = foo(d+(1:d),:);
            
            % The discrete-time dynamical model
            if (j==1)
                A  = AA(1:d,1:d,j);
                Q  = Pinf - A*Pinf*A';
                PP = A*P*A' + Q;
            end
            
            % The derivatives of A and Q
            dA = AA(d+1:end,1:d,j);
            dAPinfAt = dA*Pinf*A';
            dQ = dPinf(:,:,j) - dAPinfAt - A*dPinf(:,:,j)*A' - dAPinfAt';
            
            % The derivatives of P
            dAPAt = dA*P*A';
            dP(:,:,j) = dAPAt + A*dP(:,:,j)*A' + dAPAt' + dQ;
        end
        
        % Set predicted m and P
        m = mm;
        P = PP;
        
        % Start the Kalman filter update step and precalculate variables
        S = H*P*H' + R;
        [LS,notposdef] = chol(S,'lower');
        
        % If matrix is not positive definite, add jitter
        if notposdef>0
            jitter = jitterSigma2*diag(rand(size(S,1),1));
            [LS,notposdef] = chol(S+jitter,'lower');
            
            % Return nan to the optimizer
            if notposdef>0
                varargout = {nan*edata,nan*gdata};
                return;
            end
        end
        
        % Continue update
        HtiS = H'/LS/LS';
        iS   = eye(size(S))/LS/LS';
        K    = P*HtiS;
        v    = yall(k) - H*m;
        vtiS = v'/LS/LS';
        
        % Loop through all parameters (Kalman filter update step derivative)
        for j=1:nparam
            
            % Innovation covariance derivative
            dS = H*dP(:,:,j)*H';% + dR(:,:,j);
            
            % Evaluate the energy derivative for j (optimized from above)
            gdata(j) = gdata(j) ...
                + .5*sum(iS(:).*dS(:)) ...
                - .5*(H*dm(:,j))*vtiS' ...
                - .5*vtiS*dS*vtiS'     ...
                - .5*vtiS*(H*dm(:,j));
            
            % Kalman filter update step derivatives
            dK        = dP(:,:,j)*HtiS - P*HtiS*dS/LS/LS';
            dm(:,j)   = dm(:,j) + dK*v - K*H*dm(:,j);
            dKSKt     = dK*S*K';
            dP(:,:,j) = dP(:,:,j) - dKSKt - K*dS*K' - dKSKt';
            
        end
        
        % Evaluate the energy
        edata = edata + .5*size(S,1)*log(2*pi) + sum(log(diag(LS))) + .5*vtiS*v;
        
        % Finish Kalman filter update step
        m = m + K*v;
        P = P - K*S*K';
        
    end

    % Account for log-scale
    gdata = gdata.*exp(w);
    
    % Return negative log marginal likelihood and gradient
    varargout = {edata,gdata};

  end
  
  
  
  
  
  