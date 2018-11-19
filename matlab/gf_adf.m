function [varargout] = gf_adf(w,x,y,ss,mom,xt)
% GF_ADF - Solve GP model by single-sweep EP (ADF)
%
% Syntax:
%   [...] = gf_adf(w,x,y,k,xt)
%
% In:
%   w     - Log-parameters (nu, sigma2, theta)
%   x     - Training inputs
%   y     - Training outputs
%   ss    - State space model function handle, [F,L,Qc,...] = @(x,theta) 
%   mom   - Moment calculations for ADF
%   xt    - Test inputs (default: empty)
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
%   Covft - Predicted joint covariance matrix (not used here)
%   lb    - 95% confidence lower bound
%   ub    - 95% confidence upper bound
%
% Description:
%   Consider the following GP model:
%
%       f ~ GP(0,k(x,x')),
%     y_i ~ p(y_i | f(x_i)),  i=1,2,...,n,
%
%   where k(x,x') is the prior covariance function. The state space model 
%   giving you linear time complexity in handling the latent function
%   is specified by the function handle 'ss' such that it returns the 
%   state space model matrices
%
%     [F,L,Qc,H,Pinf,dF,dQc,dPinf] = ss(x,theta),
%
%   where theta holds the hyperparameters. See the paper [1] for details.
%   This code is assuming a stationary covariance function even though the
%   methodology per se does not require it.
%     The non-Gaussian likelihood is dealt with using single-sweep EP, also
%   known as assumed density filtering (ADF). See paper [2] for details.
%
%   NOTE: This code is proof-of-concept, not optimized for speed.
%
% References:
%
%   [1] Simo Sarkka, Arno Solin, Jouni Hartikainen (2013).
%       Spatiotemporal learning via infinite-dimensional Bayesian
%       filtering and smoothing. IEEE Signal Processing Magazine,
%       30(4):51-61.
%   [2] Hannes Nickisch, Arno Solin, and Alexander Grigorievskiy (2018). 
%       State space Gaussian processes with non-Gaussian likelihood. 
%       International Conference on Machine Learning (ICML). 
%
% Copyright:
%   2014-2018   Arno Solin
%
%  This software is distributed under the GNU General Public
%  License (version 3 or later); please refer to the file
%  License.txt, included with the software, for details.

%% Check defaults

  % Is there test data
  if nargin < 6, xt = []; end
   
  
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
  
  % Form the state space model
  [F,L,Qc,H,Pinf,dF,dQc,dPinf] = ss(x,param(1:end));
  
  
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
    ttau = zeros(1,size(yall,1));
    tnu = zeros(1,size(yall,1));
    lZ = zeros(1,size(yall,1));
    R = zeros(1,size(yall,1));
    fs2 = zeros(1,size(yall,1));
    fmus = zeros(1,size(yall,1));
    
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
            
            % Latent marginal, cavity distribution
            fmu = H*m; W = P*H'; HPH = H*W;
            
            % Propagate moments through likelihood
            [lZ(k),dlZ,d2lZ] = mom(fmu,HPH,k);          
            
            % Perform moment matching
            ttau(k) = -d2lZ/(1+d2lZ*HPH);
            tnu(k)  = (dlZ-fmu*d2lZ)/(1+d2lZ*HPH);
            
            % This is the equivalent measurement noise
            R(k) = -(1+d2lZ*HPH)/d2lZ;
            
            fs2(k) = HPH;
            fmus(k) = fmu;
            
            % Enforce positivity->lower bound ttau by zero
            ttau(k) = max(ttau(k),0);            
            if ttau(k)==0
              warning('Moment matching hit bound.')
              z = ttau(k)*HPH+1;
              K = ttau(k)*W/z;
              v = ttau(k)*fmu - tnu(k);
              m = m - W*v/z;
              P = P - K*W';
            else
              K = W/(HPH+1/ttau(k));
              v = tnu(k)/ttau(k) - fmu;
              m = m + K*v;
              P = P - K*H*P;
            end
        end
        
        % Store estimate
        MS(:,k)   = m;
        PS(:,:,k) = P;
        
    end
    
    % Output debugging info
    if nargout>5
      out.tnu = tnu;
      out.ttau = ttau;
      out.lZ = lZ;
      out.R = R;
      out.MF = MS;
      out.PF = PS;
      out.fs2 = fs2;
      out.fmu = fmus;
    end
    
    % ### Backward smoother
      
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
          jitterSigma2 = 1e-4;
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
    
    % Estimate the joint covariance matrix if requested
    if nargout > 2
      
      % Allocate space for results
      %Covft = zeros(size(PS,3));
      Covft = [];
          
      % Lower triangular
      %{
      for k = 1:size(PS,3)-1
        GSS = GS(:,:,k);
        for j=1:size(PS,3)-k
          Covft(k+j,k) = H*(GSS*PS(:,:,k+j))*H';
          GSS = GSS*GS(:,:,k+j);
        end
      end
      %}
    
    end
    
    out.MS = MS;
    out.PS = PS;
  
    % These indices shall remain to be returned
    MS = MS(:,return_ind);
    PS = PS(:,:,return_ind);
    
    % Return mean
    Eft = H*MS;
    
    % Return variance
    if nargout > 1
        Varft = zeros(size(H,1),size(H,1),size(MS,2));
        for k=1:size(MS,2)
            Varft(:,:,k)  = H*PS(:,:,k)*H';
        end
    end
    
    % Return values
    varargout = {Eft(:),Varft(:)};
 
    % Also return joint covariance and upper/lower 95% bounds
    if nargout > 3
        
        % Join upper triangular and variances
        %if ~filteronly
        %    Covft = Covft(return_ind,return_ind);
        %    Covft = Covft+Covft'+diag(Varft(:));
        %else
        %    Covft = [];
        %end
        
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
    
    ttau = zeros(1,size(yall,1));
    tnu = zeros(1,size(yall,1));
    z = zeros(1,size(yall,1));
    lZ = zeros(1,size(yall,1));
    dz =  zeros(size(yall,1),nparam);
        
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
        
        % Latent marginal, cavity distribution
        fmu = H*m; W = P*H'; fs2 = H*W;

        % Propagate moments through likelihood
        [lZ(k),dlZ,d2lZ] = mom(fmu,fs2,k);
                
        % Perform moment matching
        ttau(k) = -d2lZ/(1+d2lZ*fs2);
        tnu(k)  = (dlZ-fmu*d2lZ)/(1+d2lZ*fs2);
        
        % Enforce positivity->lower bound ttau by zero
        ttau(k) = max(ttau(k),0);
      
        % Gauss: H*m~N(r,W^-1) ttau = W, tnu = W*r, r=y-m(x)
        z(k) = ttau(k)*fs2+1;
        K = ttau(k)*W/z(k);
        v = ttau(k)*fmu - tnu(k);
        
        % Do update
        m = m-W*v/z(k);
        P = P-K*W';
        
    end
    
    % Sum up the lik
    edata = -sum(lZ);
    
    % Account for log-scale
    gdata = gdata.*exp(w);
    
    % Return negative log marginal likelihood and gradient
    varargout = {edata,gdata};

  end
  
  
  
  
  
  