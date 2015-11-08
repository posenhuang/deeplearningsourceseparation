function varargout=bss_decomp_filt(varargin)

% decompose an estimated source into target/interference/noise/artefacts components, assuming the admissible distortion is a pure time-invariant filter.
% components, assuming the admissible distortion is a pure time-invariant
% filter.
%
% Usage:
%
% [s_target,e_interf[,e_noise],e_artif]=bss_decomp_filt(se,index,S[,N],L)
%
% Input:
%   - se: row vector of length T containing the estimated source,
%   - index: points which component of S se has to be compared to,
%   - S: n x T matrix containing the original sources,
%   - N: m x T matrix containing the noise on the observations (if any).
%   - L: the number of lags of the allowed filter
%
% Output:
%   - s_target: row vector of length T containing the target source(s)
%   contribution,
%   - e_interf: row vector of length T containing the interferences
%   contribution,
%   - e_noise: row vector of length T containing the noise contribution (if
%   any),
%   - e_artif: row vector of length T containing the artifacts
%   contribution.
%
% Developers:  - Cedric Fevotte (fevotte@tsi.enst.fr) - Emmanuel Vincent
% (emmanuel.vincent@irisa.fr) - Remi Gribonval (remi.gribonval@irisa.fr)

se=varargin{1}; index=varargin{2}; S=varargin{3};
        
switch nargin
    case 4
    N=[];
    L=varargin{4};
    case 5
    N=varargin{4};
    L=varargin{5};
    otherwise
    disp('Wrong number of arguments.')
end
    
[ne,Te]=size(se);
[n,T]=size(S);

%%%%%%%%%% WARNINGS %%%%%%%%%%%%%
switch isempty(N)
    case 1
        if n>T | ne>Te, disp('Watch out: signals must be in rows.'), return; end        
        if ne~=1, disp('Watch out: se must contain only one row.'), return; end
        if T~=Te, disp('Watch out: se and S have different lengths.'), return; end        
    case 0
        [m,Tm]=size(N);        
        if n>T | ne>Te | m>Tm, disp('Watch out: signals must be in rows.'), return; end        
        if ne~=1, disp('Watch out: se must contain only one row.'), return; end
        if T~=Te, disp('Watch out: S and Se have different lengths.'), return; end        
        if T~=Tm, disp('Watch out: N, S and Se have different lengths.'), return; end        
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create the space of target source(s)
target_space=bss_make_lags(S(index,:),L); 
% Create the space of sources
sources_space=bss_make_lags(S,L);
% Create the noise space
noise_space=bss_make_lags(N,L);

s_target=zeros(1,T);
e_interf=zeros(1,T);
e_artif=zeros(1,T);
if isempty(noise_space)==0, e_noise=zeros(1,T); end

%%% Target source(s) contribution %%%
s_target = bss_proj(se,target_space);

%%% Interferences contribution %%%
P_S_se = bss_proj(se,sources_space);
e_interf = P_S_se - s_target;

switch isempty(noise_space)
    case 1 % No noise
        %%% Artifacts contribution %%%
        e_artif= se - P_S_se;
        
        %%% Output %%%
        varargout{1}=s_target;
        varargout{2}=e_interf;
        varargout{3}=e_artif;
        
    case 0 % Noise
        %%% Noise contribution %%%
        P_SN_se= bss_proj(se,[sources_space;noise_space]);
        e_noise=P_SN_se-P_S_se;
        
        %%% Artifacts contribution %%%  
        e_artif=se-P_SN_se;
        
        %%% Output %%%
        varargout{1}=s_target;
        varargout{2}=e_interf;
        varargout{3}=e_noise;
        varargout{4}=e_artif;        
end        
