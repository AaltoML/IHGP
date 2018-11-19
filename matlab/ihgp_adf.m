function [varargout] = ihgp_adf(w,x,y,ss,mom,xt)
% IHGP_ADF - Infinite-horizon GP with single-sweep EP (ADF)
%
% Syntax:
%   [...] = ihgp_adf(w,x,y,k,xt)
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
%   Covft - Predicted joint covariance matrix (disabled output)
%   lb    - 95% confidence lower bound
%   ub    - 95% confidence upper bound
%
% Description:
%   Consider the general GP modelling problem:
%
%     f(t) ~ GP(0,k(x,x')),
%      y_i ~ p(y_i | f(x_i)),  i=1,2,...,n,
%
%   where k(x,x') = k_theta(x,x'). This function performs assumed density
%   filtering (ADF) / single-sweep expectation propagtaion for dealing with
%   non-Gaussian likelihoods (observation models).
%
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
%       Information Processing Systems (NIPS). Montreal, Canada. 
%
% Copyright:
%   2018 Arno Solin
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
  
  % Check
  if std(diff(xall))>1e-12
      error('This function only accepts equidistant inputs for now.')
  end
  
  
%% Set up model

  % Log transformed parameters
  param = exp(w);

  % Extract values
  n = numel(x);
  
  % Form the state space model
  [F,L,Qc,H,Pinf,dF,dQc,dPinf] = ss(x,param(1:end));

  
%% Solve a bunch of DAREs

    % Set A and Q (this part only works for stationary covariance functions)
    dt = xall(2)-xall(1);
    A = expm(F*dt);
    Q = Pinf - A*Pinf*A';
    Q = (Q+Q')/2;
        
    % Set up interpolation setup
    ro = logspace(-2,4,32)';
    PPlisto = nan(numel(ro),numel(Q));
    for j=1:numel(ro)
      try
        [PP,~,~,report] = dare(A',H',Q,ro(j));
        PPlisto(j,:) = PP(:)';
      catch
        warning('Failed forward DARE calculation %i.',j)
        if false
          varargout = {nan,nan*w};
          return
        else
          ro(j) = nan;
        end          
      end
    end
    PPlisto(isnan(ro),:)=[]; ro(isnan(ro)) = [];
    
    % Interpolation for lookup (cubic)
    r = logspace(-2,4,200)';
    W = apxGrid('interp',{ro},r,3);
    PPlist = W*PPlisto;
    
    % Output
    if nargout>5
      out.r = r;
      out.ro = ro;
      out.PPlisto = PPlisto;
      out.PPlist = PPlist;
    end
    
%% Prediction of test inputs (filtering and smoothing)

  % Check that we are predicting
  if ~isempty(xt)

    % Set initial state
    m = zeros(size(F,1),1);
    P = Pinf;
    
    % Allocate space for results
    MS = zeros(size(m,1),size(yall,1));
    PS = zeros(size(m,1),size(m,1),size(yall,1));
    ttau = zeros(1,size(yall,1));
    tnu = zeros(1,size(yall,1));
    z = zeros(1,size(yall,1));
    lZ = zeros(1,size(yall,1));  
    R = zeros(1,size(yall,1));
    
    % ### Forward filter
    
    % The filter recursion
    for k=1:numel(yall)
        
        % Look-up
        if k>1
          [~,ind] = min(abs(r-R(k-1)));
          PP = reshape(PPlist(ind,:),size(A));
        else
          PP = Pinf;
        end
        
        % Latent marginal, cavity distribution
        fmu = H*A*m; W = PP*H'; HPH = H*W;
        
        % Propagate moments through likelihood
        [lZ(k),dlZ,d2lZ] = mom(fmu,HPH,k);
        
        % Perform moment matching
        ttau(k) = -d2lZ/(1+d2lZ*HPH);
        tnu(k)  = (dlZ-fmu*d2lZ)/(1+d2lZ*HPH);
        
        % Enforce positivity->lower bound ttau by zero
        ttau(k) = max(ttau(k),0);
        
        % This is the equivalent measurement noise
        R(k) = -(1+d2lZ*HPH)/d2lZ;
        
        yall(k) = tnu(k)./ttau(k);
        
        % Deal with special cases
        if ttau(k)==0
            warning('Moment matching hit bound!')
            R(k) = inf;
            m = A*m;
            P = PP;
        else
            
            % Gain
            K = W/(HPH+R(k));
            
            % Precalculate
            AKHA = A-K*H*A;
            
            % The stationary filter recursion
            m = AKHA*m + K*yall(k);             % O(m^2)
            P = PP-K*R(k)*K';
            
        end
        
        % Store estimate
        MS(:,k)   = m;
        PS(:,:,k) = P;
        
    end
    
    % Output debugging info
    if nargout>5
      out.tnu = tnu;
      out.ttau = ttau;
      out.z = z;
      out.lZ = lZ;
      out.R = R;
      out.MF = MS;
      out.PF = PS;
    end
    
    % ### Backward smoother
    
    % Set up interpolation setup
    PGlist = [];
    for j=1:numel(ro)
        PP = reshape(PPlisto(j,:),size(P));
        S  = H*PP*H'+ro(j);
        K = PP*H'/S;
        P = PP-K*ro(j)*K';
        
        % Calculate cholesky
        [L,notpositivedefinite] = chol(A*P*A'+Q,'lower');
        if notpositivedefinite>0
            [V,D]=eig(A*P*A'+Q); ind=diag(D)>0; APAQ = V(:,ind)*D(ind,ind)*V(:,ind)';
            L = cholcov(APAQ)';
        end
        
        % Solve the associated DARE (with some stabilization tricks)
        G = P*A'/L'/L;
        QQ = P-G*(PP)*G'; QQ = (QQ+QQ')/2;
        [V,D]=eig(QQ); ind=diag(D)>0; QQ = V(:,ind)*D(ind,ind)*V(:,ind)';
        try
            PS2 = dare(G',0*G,QQ);
        catch
            PS2 = P*0;
            ro(j) = nan;
            fprintf('Failed smoother DARE %i\n',j)
        end
        PGlist(j,:) = [PS2(:)' G(:)']; %#ok
    end
    PGlist(isnan(ro),:)=[]; ro(isnan(ro)) = [];
    
    % Interpolate
    W = apxGrid('interp',{ro},r,3);
    PGlist = W*PGlist;
    
    % Final state
    try
    
        % Final state covariance 
        P = PS(:,:,end);
        
        % The gain and covariance
        [L,notpositivedefinite] = chol(A*P*A'+Q,'lower');
        if notpositivedefinite>0
            [V,D]=eig(A*P*A'+Q); ind=diag(D)>0; APAQ = V(:,ind)*D(ind,ind)*V(:,ind)';
            L = cholcov(APAQ)';
        end
        G = P*A'/L'/L;
        
        QQ = P-G*(A*P*A'+Q)*G'; QQ = (QQ+QQ')/2;
        [V,D]=eig(QQ); ind=diag(D)>0; QQ = V(:,ind)*D(ind,ind)*V(:,ind)';
        PS2 = dare(G',0*G,QQ);
        PS(:,:,end) = PS2;
        
    catch
        if isinf(R(k))
          ind = numel(r);
        else
          [~,ind] = min(abs(r-R(k)));
        end
        PG = PGlist(ind,:);
        PS(:,:,end) = reshape(PG(1:end/2),size(P)); 
        %G = reshape(PG(end/2+1:end),size(P));
    end
          
    % Rauch-Tung-Striebel smoother
    for k=size(MS,2)-1:-1:1
        
        % Look-up
        if isinf(R(k))
          ind = numel(r);
        else
          [~,ind] = min(abs(r-R(k)));
        end
        PG = PGlist(ind,:);
        P = reshape(PG(1:end/2),size(P));
        G = reshape(PG(end/2+1:end),size(P));

        % Backward iteration
        m = MS(:,k) + G*(m-A*MS(:,k));       % O(m^2)
        
        % Store estimate
        MS(:,k)   = m;
        PS(:,:,k) = P;
        
    end
    
    % Debugging information
    if nargout>5
        out.MS = MS;
        out.PS = PS;
    end
      
    % Estimate the joint covariance matrix if requested
    if nargout > 2
      
      % Allocate space for results
      Covft = []; %zeros(size(PS,3));
          
      % Lower triangular (this output is not used)
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
    gdata = zeros(1,nparam);
    
    % Set up
    m  = zeros(d,1);    
    ttau = zeros(1,size(yall,1));
    tnu = zeros(1,size(yall,1));
    lZ = zeros(1,size(yall,1));
    
    % Loop over all observations
    for k=1:steps
             
        % Look-up
        if k>1
          [~,ind] = min(abs(r-R(k-1)));
          PP = reshape(PPlist(ind,:),size(A));
        else
          PP = Pinf;
        end
        
        % Latent marginal, cavity distribution
        fmu = H*A*m; W = PP*H'; HPH = H*W;
        
        % Propagate moments through likelihood
        [lZ(k),dlZ,d2lZ] = mom(fmu,HPH,k);
        
        % Perform moment matching
        ttau(k) = -d2lZ/(1+d2lZ*HPH);
        tnu(k)  = (dlZ-fmu*d2lZ)/(1+d2lZ*HPH);
        
        % Enforce positivity->lower bound ttau by zero
        ttau(k) = max(ttau(k),0);
        
        % This is the equivalent measurement noise
        R(k) = -(1+d2lZ*HPH)/d2lZ;
        
        yall(k) = tnu(k)./ttau(k);
        
        % Deal with special cases
        if ttau(k)==0
            warning('Moment matching hit bound!')
            R(k) = inf;
            m = A*m;
        else
            
            % Gain
            K = W/(HPH+R(k));

            % Precalculate
            AKHA = A-K*(H*A);
            
            % The stationary filter recursion
            m = AKHA*m + K*yall(k);             % O(m^2)
            
        end
        
        
    end

    % Sum up
    edata = -sum(lZ);
    
    % Account for log-scale
    gdata = gdata.*exp(w);
    
    % Return negative log marginal likelihood and gradient
    varargout = {edata,gdata};

  end
  