function [ ] = save_callback( theta, info, state, eI, varargin)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% save model while minfunc is running

if mod(info.iteration, 50) == 0    
    if isfield(eI, 'iterStart')
      info.iteration = info.iteration+eI.iterStart;
    end
    saveName = sprintf('%smodel_%d.mat',eI.saveDir,info.iteration);
    save(saveName, 'theta', 'eI',  'info');
end;

end

