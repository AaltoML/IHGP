function [varargout] = ihgpr(w,x,y,ss,xt,filteronly,opt,w0)
% IHGPR - Infinite-horizon GP regression
%
% Syntax:
%   [...] = ihgpr(w,x,y,k,xt)
%
% In:
%   w     - Log-parameters (sigma2, theta)
%   x     - Training inputs
%   y     - Training outputs
%   ss    - State space model function handle, [F,L,Qc,...] = @(x,theta) 
%   xt    - Test inputs (default: empty)
%   filteronly - Run only filter (default: false)
%   opt   - Which parameters to optimize
%   w0    - Log-parameters (all / default values)
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
%   Covft - Predicted joint covariance matrix (disabled output)
%   lb    - 95% confidence lower bound
%   ub    - 95% confidence upper bound
%
% Description:
%   Consider the following GP regression problem:
%
%       f ~ GP(0,k(x,x')),
%     y_i = f(x_i) + e_i,  i=1,2,...,n,
%
%   where k(x,x') = k_theta(x,x') and e_i ~ N(0,sigma2).
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
%   [1] Arno Solin, James Hensman, and Richard E. Turner (2018). 
%       Infinite-horizon Gaussian processes. Advances in Neural 
%       Information Processing Systems (NIPS). Montréal, Canada. 
%
% Copyright:
%   2018 Arno Solin
%
%  This software is distributed under the GNU General Public
%  License (version 3 or later); please refer to the file
%  License.txt, included with the software, for details.
%

%% Check defaults

  % Is there test data
  if nargin < 5, xt = []; end
  
  % Is filteronly set
  if nargin < 6 || isempty(filteronly), filteronly = false; end
  
  % Is filteronly set
  if nargin > 6 && ~isempty(opt)
    w0(opt) = w;
    w = w0;
  else
    opt = true(size(w));
  end
  
  
%% Figure out the correct way of dealing with the data

  % Combine observations and test points
  xall = [x(:); xt(:)];
  yall = [y(:); nan(numel(xt),1)];
    
  % Make sure the points are unique and in ascending order
  [xall,sort_ind,return_ind] = unique(xall,'first');
  yall = yall(sort_ind);
  
  % Only return test indices
  return_ind = return_ind(end-numel(xt)+1:end);
  
  % Check
  if std(diff(xall))>1e-12
      error('This function only accepts equidistant inputs for now.')
  end
  
  
%% Set up model

  % Log transformed parameters
  param = exp(w);

  % Extract values
  d      = numel(x);
  sigma2 = param(1);
  
  % Form the state space model
  try
    [F,L,Qc,H,Pinf,dF,dQc,dPinf] = ss(x,param(2:end));
  catch
    if isempty(xt)
      % On failure, return nan to indicate a bad point
      varargout = {nan,nan*param};  
      return          
    else
      error('Problems with state space model.')
    end
  end
  
  % Concatenate derivatives
  dF    = cat(3,zeros(size(F)),dF);
  dQc   = cat(3,zeros(size(Qc)),dQc);
  dPinf = cat(3,zeros(size(Pinf)),dPinf);
  dR    = zeros(1,1,numel(param)); dR(1) = 1;  
  
  
%% Do the stationary stuff

  % Parameters (this assumes the prior covariance function to be stationary)
  dt = xall(2)-xall(1);
  A = expm(F*dt);
  Q = Pinf - A*Pinf*A'; Q = (Q+Q')/2;
  R = sigma2;
   
  % Solve the Riccati equation for the predictive state covariance
  try
    [PP,~,~,report] = dare(A',H',Q,R);
  catch
    if isempty(xt)
      % On failure, return nan to indicate a bad point
      varargout = {nan,nan*param};  
      return
    else
      error('Unstable DARE solution!')  
    end
  end      
      
  % Check the riccati result
  if report == -1
    error('The Symplectic matrix has eigenvalues on the unit circle')
  end
      
  % Innovation variance
  S = H*PP*H'+R;
  
  % Stationary gain
  K = PP*H'/S;
  
  % Precalculate
  AKHA = A-K*H*A;
  
  
%% Prediction of test inputs (filtering and smoothing)

  % Check that we are predicting
  if ~isempty(xt)

    % Set initial state
    m = zeros(size(F,1),1);
    PF = PP - K*H*PP;
    
    % Allocate space for results
    MS = zeros(size(m,1),size(yall,1));
    PS = zeros(size(m,1),size(m,1),size(yall,1));
    
    
    % ### Forward filter
    
    % The filter recursion
    for k=1:numel(yall)
        
        if ~isnan(yall(k))  
        
          % The stationary filter recursion
          m = AKHA*m + K*yall(k);             % O(m^2)
        
          % Store estimate
          MS(:,k)   = m;
          PS(:,:,k) = PF; % This is the same for all points
        
        else
     
          m = A*m;       
          MS(:,k)   = m;
          PS(:,:,k) = Pinf;
          
        end
    end
      
    % ### Backward smoother

    GS = [];
    
    % Should we run the smoother?
    if ~filteronly

      % The gain and covariance
      [L,notpositivedefinite] = chol(PP,'lower');      
      G = PF*A'/L'/L;
      
      % Solve the Riccati equation
      QQ = PF-G*(PP)*G'; QQ = (QQ+QQ')/2;
      P = dare(G',0*G,QQ);
      PS(:,:,end) = P;
      
      % Allocate space for storing the smoother gain matrix
      GS = zeros(size(F,1),size(F,2),size(yall,1));
      
      % Rauch-Tung-Striebel smoother
      for k=size(MS,2)-1:-1:1

        % Backward iteration
        m = MS(:,k) + G*(m-A*MS(:,k));       % O(m^2)

        % Store estimate
        MS(:,k)   = m;
        PS(:,:,k) = P;
        GS(:,:,k) = G;

      end
      
    end
    
    % Output debug information
    out.K = K;
    out.G = G;
    out.S = S;
    out.P = P;
    out.PP = PP;
    
    % Estimate the joint covariance matrix if requested
    if nargout > 2 && ~filteronly
      
      % Allocate space for results
      Covft = [];%zeros(size(PS,3));
      
      %{
      % Lower triangular
      for k = 1:size(PS,3)-1
        GSS = GS(:,:,k);
        for j=1:size(PS,3)-k
          Covft(k+j,k) = H*(GSS*PS(:,:,k+j))*H';
          GSS = GSS*GS(:,:,k+j);
        end
      end
      %}
      
    end
  
    % These indices shall remain to be returned
    MS = MS(:,return_ind);
    PS = PS(:,:,return_ind);
    
    % Return mean
    Eft = H*MS;
    
    % Return variance
    Varft = zeros(size(H,1),size(H,1),size(MS,2));
    for k=1:size(MS,2)
      Varft(:,:,k)  = H*PS(:,:,k)*H';
    end
    
    % Return values
    varargout = {Eft(:),Varft(:),out};
 
    % Also return joint covariance and upper/lower 95% bounds
    if nargout > 3
        
        % Join upper triangular and variances
        if ~filteronly
            %Covft = Covft(return_ind,return_ind);
            %Covft = Covft+Covft'+diag(Varft(:));
        else
            Covft = [];
        end
        
        % The bounds
        lb = Eft(:) - 1.96*sqrt(Varft(:));
        ub = Eft(:) + 1.96*sqrt(Varft(:));
        varargout = {Eft(:),Varft(:),Covft,lb(:),ub(:),out};
        
    end

  end
  
%% Evaluate negative log marginal likelihood and its gradient

  if isempty(xt)

    % Size of inputs
    d      = size(F,1);
    nparam = numel(param);
    
    % Allocate space for derivative matrices
    dA    = zeros(d,d,nparam);
    dPP   = zeros(d,d,nparam);
    dAKHA = zeros(d,d,nparam);
    dK    = zeros(d,1,nparam);
    dS    = zeros(1,1,nparam);
    HdA   = zeros(d,nparam);
    
    % Precalculate Z and B
    Z = zeros(d);
    B = A*K; % = A*PP*H'/(H*PP*H'+R);
    
    % Evaluate all derivatives
    for j=1:numel(param)
      
      % The first matrix for the matrix factor decomposition
      FF = [ F        Z;
            dF(:,:,j) F];
        
      % Solve the matrix exponential
      AA = expm(FF*dt);
      dA(:,:,j) = AA(d+1:end,1:d);
      dQ = dPinf(:,:,j) - dA(:,:,j)*Pinf*A' - A*dPinf(:,:,j)*A' - A*Pinf*dA(:,:,j)';
      dQ = (dQ+dQ')/2;
      
      % Precalculate C
      C = dA(:,:,j)*PP*A' + A*PP*dA(:,:,j)' - dA(:,:,j)*PP*H'*B' - ...
          B*H*PP*dA(:,:,j)' + B*dR(:,:,j)*B' + dQ;
      C = (C+C')/2;
      
      % Solve dPP
      try
        dPP(:,:,j) = dare((A-B*H)',zeros(d,d),C);
      catch
        % On failure, return nan to indicate a bad point
        varargout = {nan,nan*param};
        return        
      end
      
      % Evaluate dS and dK
      dS(:,:,j) = H*dPP(:,:,j)*H' + dR(:,:,j);
      dK(:,:,j) = dPP(:,:,j)*H'/S - PP*H'*(S\(H*dPP(:,:,j)*H'+dR(:,:,j))/S);
      dAKHA(:,:,j) = dA(:,:,j) - dK(:,:,j)*H*A - K*H*dA(:,:,j);
      HdA(:,j) = (H*dA(:,:,j))';
      
    end
    
    % Reshape for vectorization
    dAKHAp = reshape(permute(dAKHA,[1 3 2]),[],d);
    dKp = reshape(dK,[],nparam);
    
    % Size of inputs
    steps = numel(yall);
    m = zeros(d,1);
    dm = zeros(d,nparam);
            
    % Allocate space for results
    edata = .5*log(2*pi)*steps + .5*log(S)*numel(x);
    gdata = .5*steps*dS(:)'/S;
    
    % Loop over all observations
    for k=1:steps
      
      if ~isnan(y(k))  
        
      % Innovation mean
      v = y(k) - H*A*m;
      
      % Marginal likelihood (approximation)
      edata = edata + .5*v^2/S; %.5*sum((v/cS).^2);
      
      % The same as above without the loop
      dv = -m'*HdA - H*A*dm;
      gdata = gdata + v*dv/S - .5*v^2*dS(:)'/S^2;
      dm = AKHA*dm + dKp*y(k);
      dm(:) = dm(:) + dAKHAp*m;
      
      % The stationary filter recursion
      m = AKHA*m + K*y(k);
      
      else
        for j=1:nparam
          dm(:,j) = A*dm(:,j) + dA(:,:,j)*m;
        end
        m = A*m;
      end
      
    end
    
    % Account for log-scale
    gdata = gdata.*exp(w);
    
    % Return correct number of parameters
    gdata = gdata(opt);
    
    % Return negative log marginal likelihood and gradient
    varargout = {edata,gdata};

  end
  
