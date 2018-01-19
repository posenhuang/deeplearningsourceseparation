function run_test_single_model
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
%% Given a model, evaluate the performance.
    baseDir = '../../';
    addpath([baseDir, filesep, 'codes']);
    addpath([baseDir, filesep, 'tools', filesep,'bss_eval']);
    addpath([baseDir, filesep, 'tools', filesep,'bss_eval_3']);
    addpath([baseDir, filesep, 'tools', filesep,'labrosa']);
    addpath(['Data_with_dev']);

    ModelPath=[baseDir, filesep, 'codes',filesep, 'timit', filesep, 'model_demo'];

    global SDR;

    SDR.deviter=0;   SDR.devmax=0;   SDR.testmax=0;
    SDR.devsar=0; SDR.devsir=0; SDR.testsar=0; SDR.testsir=0;

    j=10;

    % Load model
    load([ModelPath, filesep, 'model_', num2str(j),'.mat']);
    eI.writewav=1;
    eI.bss3=1;
    eI.DataPath=[baseDir, filesep, 'codes', filesep, 'timit', ...
        filesep, 'Wavfile', filesep];
    eI.saveDir = [baseDir, filesep, 'codes', filesep, 'timit', ...
        filesep, 'model_demo', filesep, 'results', filesep];
  
    test_timit_general_kl_recurrent(eI.modelname, theta, eI, 'testall', info.iteration);
end

