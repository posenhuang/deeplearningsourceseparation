function [PY_x coeff] = bss_tvproj(x,Y,tvshape,tvstep)

% compute the orthogonal projection of x onto the space of shifted windowed versions of the row(s) of Y
%
% Usage: [PY_x coeff] = bss_tvproj(x,Y,tvshape,tvstep)
%
% Input:
%   - x: row vector of length T corresponding to the signal to be projected,
%   - Y: vector or matrix of length T with n_rows rows which windowed rows span the projection space.
%   - tvshape : row vector of length V corresponding to the window applied
%   to Y to define the projection space
%   - tvstep  : number of samples between two adjacent windows
%
% Ouput:
%   - PY_x: row vector of length T containing the orthogonal projection of
%   x onto the range of the shifted windowed versions of the row(s) of Y.
%   - coeff : matrix with n_rows rows and n_frames columns containing the coefficients 
%   of the projection 
%
% Developers:  - Cedric Fevotte (fevotte@tsi.enst.fr) - Emmanuel Vincent
% (emmanuel.vincent@irisa.fr) - Remi Gribonval (remi.gribonval@irisa.fr)

% 1. Padd the signals with zeros on the left and the right 
V        = size(tvshape,2); % the size of the window
NOVERLAP = V-tvstep;        % convert the hop size into a number of overlapping samples
NPadLeft = tvstep;
NPadRight= 2*V;
x = [zeros(size(x,1),NPadLeft) x zeros(size(x,1),NPadRight)];
Y = [zeros(size(Y,1),NPadLeft) Y zeros(size(Y,1),NPadRight)];

% 2. Decompose Y into frames using tvshape as a window

[Y_frames frames_index] = bss_make_frames(Y,tvshape,NOVERLAP); 

% 3. Y_frames is a 3-D array of size n_frames x V x n_rows
% we reshape it to get a 2-D array of size (n_frames x n_rows) x V

[n_frames V1 n_rows]= size(Y_frames); % The frames are Y_frames(f,:,n) 
Y_frames = permute(Y_frames,[2 1 3]); 
% Y_frames is now V x n_frames x nrows, the frames are Y_frames(:,f,n)

Y_frames = reshape(Y_frames,V,n_frames*n_rows); 
% Y_frames is now V x (n_frames x n_rows), the frames are Y_frames(:,k)
% with k=(n-1)*n_frames+f

Y_frames = permute(Y_frames,[2 1]);
% Y_frames is now (n_frames x n_rows) x V, the frames are Y_frames(k,:)
% with k=(n-1)*n_frames+f

% 4. Compute the inner products between x and the frames of the row(s) of Y

x_frames  = bss_make_frames(x,ones(1,V),NOVERLAP); 
% x_frames is an array n_frames x V x 1 which frames are x_frames(n,:)

ip = zeros(n_frames*n_rows,1);
for n=1:n_rows % loop on rows of Y
    idxrange = ((n-1)*n_frames)+(1:n_frames); % the index of the frames in Y_frames corresponding to the selected row
    ip(idxrange) = sum(x_frames .* Y_frames(idxrange,:),2); 
end


% 5. Compute the Gram matrix, which is square of size (n_frames x n_rows) x
% (n_frames x n_rows) 
Gram = sparse(n_frames*n_rows,n_frames*n_rows);

for f=1:n_frames 
    for f1 = 1:n_frames
        % locate the range of the intersection between frames
        first = max(frames_index(f),frames_index(f1));
        last  = min(frames_index(f),frames_index(f1))+V-1;
        % if the intersection is non empty, fill it
        if (first <= last)
            nrange  = (first:last)-frames_index(f)+1;
            nrange1 = (first:last)-frames_index(f1)+1;
            idxrange  = f  + n_frames*((1:n_rows)-1);
            idxrange1 = f1 + n_frames*((1:n_rows)-1);
            Gram(idxrange,idxrange1) = Y_frames(idxrange,nrange)*Y_frames(idxrange1,nrange1)';
        end
    end
end

% 6. Apply the inverse of the Gram matrix to ip to get coeff
%coeff = full(Gram)\ip;
coeff = pinv(full(Gram))*ip;
% 7.  Reconstruct using coefficients alpha
PY_x = zeros(size(x));

for f=1:n_frames
    idxrange = f + n_frames*((1:n_rows)-1);
    trange   = frames_index(f)+(0:(V-1));
    PY_x(trange) = PY_x(trange) + coeff(idxrange)'*Y_frames(idxrange,:);
end

% 8. Truncate the projection to the support of the signal
PY_x = PY_x(:,(NPadLeft+1):(end-NPadRight));

% 9. The coefficients are a n_frames*n_rows column vector with entries 
% coeff(f+(n-1)*n_frames). We reshape it
coeff = reshape(coeff,n_frames,n_rows);
coeff = permute(coeff,[1 2]);

