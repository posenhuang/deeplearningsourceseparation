function [PY_x coeff]=bss_proj(x,Y)

% compute the orthogonal projection of x on the subspace spanned by the row(s) of Y.
%
% Usage: PY_x         = proj(x,Y)
%        [PY_x coeff] = proj(x,Y)
%
% Input:
%   - x: row vector of length T,
%   - Y: vector or matrix of length T.
%
% Ouput:
%   - PY_x: row vector of length T containing the orthogonal projection of
%   x onto the range of the rows of Y.
%   - coeff : column vector with as many rows as Y containing the
%   coefficients such that PY_x = coeff.'*Y
%
% Developers:  - Cedric Fevotte (cf269@cam.ac.uk) - Emmanuel Vincent
% (vincent@ircam.fr) - Remi Gribonval (remi.gribonval@irisa.fr)

% Gram matrix of Y
G=Y*Y';

%same as coeff=inv(conj(G))*conj(Y*x');
coeff=conj(G)\conj(Y*x');
%if the Gram matrix G is not invertible then coeff=pinv(conj(G))*conj(Y*x')
%should work, but in general it is much slower than the default code

PY_x=  coeff.'*Y;

% Same as PY_x= x*pinv(Y)*Y;