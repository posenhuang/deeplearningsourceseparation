function S_lags=bss_make_lags(S,L)

% create a matrix containing lagged versions of some signal(s).
%
% Usage: S_lags=bss_make_lags(S,L)
%
% Input:
%   - S: n x T matrix containing the input signal(s),
%   - L: number of lagged versions of the signal(s).
%
% Output:
%   - S_lags: n*L x T matrix containing lagged versions of S, S_lags(t)=
%   [s1(t) ; s1(t-1) ; ...; s1(t-L+1); ... ; sn(t) ; sn(t-1) ; ...; sn(t-L+1)]
%
% WARNINGS:
%   * S_lags is zero-padded where necessary,
%   * We use the conventions make_lags(S,0)=makes_lags(S,1)=S.
%
% Developers:  - Cedric Fevotte (fevotte@tsi.enst.fr) - Emmanuel Vincent
% (emmanuel.vincent@irisa.fr) - Remi Gribonval (remi.gribonval@irisa.fr)

[n,T]=size(S);

if L==0
    S_lags=S;
else
    
    N=n*L;
    S_lags=zeros(N,T);
    for i=1:N
        q=floor((i-1)/L);
        r=mod(i-1,L);
        B=zeros(1,r+1); B(end)=1;
        S_lags(i,:)=filter(B,1,S(q+1,:));    
    end
    
end