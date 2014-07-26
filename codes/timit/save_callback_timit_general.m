function [ ] = save_callback_timit_general( theta, info, state, eI, varargin)
% save model while minfunc is running

if mod(info.iteration, 10) == 0
    if isfield(eI, 'iterStart')
      info.iteration = info.iteration+eI.iterStart;
    end
    if ~strcmp(state,'init')
        if isfield(eI,'ioffset'),
            saveName = sprintf('%smodel_off%d_%d.mat',eI.saveDir,eI.ioffset, ...
                info.iteration);
        else
            saveName = sprintf('%smodel_%d.mat',eI.saveDir,info.iteration);
        end
        save(saveName, 'theta', 'eI',  'info');
    end

    % run evaluation
    test_timit_general_kl_recurrent(eI.modelname, theta, eI, 'iter', ...
        info.iteration);
end;

end

