function [PY_x coeff] = bss_tvproj(x,Y,tvshape,tvstep)

% compute the orthogonal projection of x onto the space of shifted windowed versions of the row(s) of Y
%
% Usage: [PY_x coeff] = tvproj(x,Y,tvshape,tvstep)
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
% Developers:  - Cedric Fevotte (cf269@cam.ac.uk) - Emmanuel Vincent
% (vincent@ircam.fr) - Remi Gribonval (remi.gribonval@irisa.fr)

% 1. Decompose Y into frames using tvshape as a window
V        = size(tvshape,2); % the size of the window
NOVERLAP = V-tvstep;        % convert the hop size into a number of overlapping samples

[Y_frames frames_index] = bss_make_frames(Y,tvshape,NOVERLAP); % Y_frames is a 3-D array
[n_frames V1 n_rows]= size(Y_frames);

% 2. Compute the inner products between x and the frames of the row(s) of Y
ip = zeros(n_frames,n_rows);
for f=1:n_frames % loop on frames
    ip(f,:) = x(frames_index(f)+(0:(V-1)) ) * reshape(Y_frames(f,:,:),V,n_rows); % columns of ip correspond to the same frame number
end
ip = reshape(ip,n_frames*n_rows,1); % now we want a single column vector to apply an inverse matrix to it

% 3. Compute the Gram matrix, which is square of size (n_frames x n_rows) x
% (n_frames x n_rows) 
Gram = zeros(n_frames*n_rows,n_frames*n_rows);
for f=1:n_frames 
    for f1 = 1:n_frames
        % locate the range of the intersection between frames
        first = max(frames_index(f),frames_index(f1));
        last  = min(frames_index(f),frames_index(f1))+V-1;
        % if the intersection is non empty, fill it
        if (first <= last)
            trange = (first:last)-frames_index(f)+1;
            trange1= (first:last)-frames_index(f1)+1;
            Gram((f-1)*n_rows+(1:n_rows),(f1-1)*n_rows+(1:n_rows)) = reshape(Y_frames(f,trange,:),length(trange),n_rows)'*reshape(Y_frames(f1,trange1,:),length(trange1),n_rows); % shall we reshape Y_frames ????
        end
    end
end

% 4. Apply the inverse of the Gram matrix to ip to get coeff
coeff = Gram\ip;
coeff = reshape(coeff,n_frames,n_rows);
% 4.  Reconstruct using coefficients alpha
PY_x = zeros(size(x));

for f=1:n_frames
    PY_x(frames_index(f)+(0:(V-1))) = PY_x(frames_index(f)+(0:(V-1))) + coeff(f,:)*reshape(Y_frames(f,:,:),V,n_rows)';
end

