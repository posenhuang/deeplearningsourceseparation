function [ ] = save_callback( theta, info, state, eI, varargin)
% save model while minfunc is running

if mod(info.iteration, 50) == 0    
    if isfield(eI, 'iterStart')
      info.iteration = info.iteration+eI.iterStart;
    end
    saveName = sprintf('%smodel_%d.mat',eI.saveDir,info.iteration);
    save(saveName, 'theta', 'eI',  'info');
end;

end

