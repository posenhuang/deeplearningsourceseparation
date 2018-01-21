function [ri,ti] = traintestindex( s, l1, l2, l3, l4)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% Create a training/testing data set based ont he leave-out info provided

ri = [];
ti = [];
for i = 1:s(1)
	for j = 1:s(2)
		for l = 1:s(3)
			for k = 1:s(4)
				if (ismember( i, l1) || isempty( l1)) && (ismember( j, l2) || isempty( l2)) ...
						&& (ismember( l, l3) || isempty( l3)) && (ismember( k, l4) || isempty( l4))
					ti(end+1) = sub2ind( s, i, j, l, k);
				elseif ~ismember( i, l1) && ~ismember( j, l2) && ~ismember( l, l3) && ~ismember( k, l4)
					ri(end+1) = sub2ind( s, i, j, l, k);
				end
			end
		end
	end
end
