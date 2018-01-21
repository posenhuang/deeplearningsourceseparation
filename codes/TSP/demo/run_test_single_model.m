function run_test_single_model
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% Given a model, evaluate the performance.
    baseDir = '../../../';
    addpath([baseDir, filesep, 'codes']);
    addpath([baseDir, filesep, 'tools', filesep, 'bss_eval']);
    addpath([baseDir, filesep, 'tools', filesep, 'bss_eval_3']);
    addpath([baseDir, filesep, 'tools', filesep, 'labrosa']);
    addpath([baseDir, filesep, 'codes', filesep, 'TSP']);
    
    ModelPath=[baseDir, filesep, 'codes',filesep, 'TSP', filesep, 'model_demo'];

    global SDR;
    SDR.deviter=0;   SDR.devmax=0;   SDR.testmax=0;
    SDR.devsar=0; SDR.devsir=0; SDR.testsar=0; SDR.testsir=0;

    j=7650;

    % Load model
    load model_RNN1_win1_h300_l2_r0_64ms_1000000_softabs_linearout_RELU_logmel_trn0_c1e-10_c0.001_bsz100000_miter10_bf50_c0_d0_7650.mat
    eI.writewav=1;
    eI.bss3=1;
    eI.DataPath=[baseDir, filesep, 'codes', filesep, 'TSP', ...
        filesep, 'Data', filesep];
    eI.saveDir = [baseDir, filesep, 'codes', filesep, 'TSP', ...
        filesep, 'demo', filesep, 'results', filesep];
    eI.CFGPath = [baseDir, filesep, 'tools', filesep, 'htk_features', filesep];
    test_TSP_general_kl_recurrent(eI.modelname, theta, eI, 'done', j);
end
