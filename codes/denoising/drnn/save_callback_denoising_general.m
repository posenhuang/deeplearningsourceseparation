function [ ] = save_callback_denoising_general( theta, info, state, eI, varargin)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% save model and run evaluation while minfunc is running

if mod(info.iteration, 100) == 0
    if isfield(eI, 'iterStart')
      info.iteration = info.iteration+eI.iterStart;
    end
    if ~strcmp(state,'init')
        if isfield(eI,'ioffset'),
            saveName = sprintf('%smodel_off%d_%d.mat',eI.saveDir,eI.ioffset, info.iteration);
        else
            saveName = sprintf('%smodel_%d.mat',eI.saveDir,info.iteration);
        end
        save(saveName, 'theta', 'eI',  'info');
    end
    % run evaluation
%       if mod(info.iteration, 20) == 0 && info.iteration>=0
%          test_mir1k_general_kl_bss3(eI.modelname, theta, eI, 'iter', info.iteration);
%       end
end;

end

