function net=train_denoising_mini(source_1, source_2, context_win, hidden_units, num_layers, isdropout, ...
    isRNN, iscleanonly, circular_step , isinputL1, MFCCorlogMelorSpectrum, ...
    framerate, pos_neg_r, outputnonlinear, opt, act, train_mode, const,  ...
    const2, isGPU, max_iter, batchsize, lbfgs_iter, grad_clip, wdecay)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% Demo Denoising training ---------------------------------------------
% context_win - context window size
% hidden_units - hidden units
% num_layers - layer number
% isdropout - 1: use dropout, 0: no dropout
% isRNN - RNN temporal connection
% iscleanonly - One output source or two
% circular_step - Circular shift step
% isinputL1 - normalize input as L1 norm = 1
% MFCCorlogMelorSpectrum - 0: MFCC, 1: logmel, 2: spectra
% framerate - feature frame rate
% pos_neg_r - discriminative training gamma parameter
% outputnonlinear - Last layer - linear or nonlinear
% softabs - soft mask obj
% act - 0: logistic, 1: tanh, 2: RELU
% const - constant for avoiding numerical problems
% const2- constant for avoiding numerical problems
% isGPU - 0: not using GPU, 1: using GPU
% train_mode - 0
% max_iter - max LBFGS iterations

rand('state',0)
randn('state',0)

%% setup paths for code. assumed this script runs in its own directory
% CHANGE baseDir to the top of code directory
baseDir= ['..',filesep, '..',  filesep];
codeDir = [baseDir,'codes', filesep];
minFuncDir = [baseDir, 'tools', filesep, 'minFunc_2012', filesep];

saveDir = [codeDir,filesep,'denoising',filesep,'discrim_joint_offset_results'];

%% add paths
addpath([baseDir, filesep,'tools', filesep,'labrosa']);
% addpath([baseDir, filesep,'tools', filesep,'bss_eval']);
addpath([baseDir, filesep,'tools', filesep,'bss_eval_2.1']);
% addpath([baseDir, filesep,'tools', filesep,'bss_eval_3']);
addpath(baseDir);
addpath(genpath(minFuncDir));
addpath(codeDir);
addpath([codeDir,'denoising']);

CFGPath=[baseDir,'tools',filesep,'htk_features', filesep];
addpath(CFGPath);

%% setup network architecture
setup_architecture; 

%% initialize weights
[stack_i, W_t_i] = initialize_weights(eI);
[theta] = rnn_stack2params(stack_i, eI, W_t_i);

%% Directory of features
eI.featInBase =baseDir;

%% load data
eI.useCache = 0;

%% setup minFunc
options.Diagnostics = 'on';
options.Display = 'iter';
% options.MaxIter = 400;

options.MaxIter = lbfgs_iter;
options.MaxFunEvals = 2500;
options.Corr = 50;
options.DerivativeCheck = 'off';
% options.DerivativeCheck = 'on';
options.outputFcn = @save_callback_denoising_general;

% eI.DataPath=[codeDir,'mir1k', filesep, 'Wavfile',filesep];
eI.iterStart=-batchsize;

%% feed data 
for ii= 1: max_iter        
    fprintf('Global iterations: %d\n', ii);
    eI.iterStart=eI.iterStart+batchsize;

    batch_cell_idx = randperm(numel(source_1), min(batchsize, numel(source_1)));
    source_1_mini = cell(1, numel(batch_cell_idx)); source_2_mini = cell(1, numel(batch_cell_idx)); 
    for jj=1:numel(batch_cell_idx)
        source_1_mini{jj} = source_1{batch_cell_idx(jj)};
        source_2_mini{jj} = source_2{batch_cell_idx(jj)};
    end
    % train_files= dir( [eI.DataPath, 'train',filesep,'*wav']);
    % 0 -- chunk, 2--no chunk
    [data_cell, targets_cell, mixture_spectrum] = ...
                formulate_data(source_1_mini, source_2_mini, eI, eI.train_mode); 

    %% BSS EVAL setting
    eI.writewav=0;

    %% run optimizer
    if isfield(eI,'cleanonly') && eI.cleanonly==1,
        % % for non joint training
        eI.isdiscrim=1;
        [theta,val]=minFunc(@drdae_discrim_obj, theta, options, eI, data_cell, ...
                            targets_cell, false, false);                          
    else   
       if isGPU==1 && strcmpi(eI.activationFn,'RELU') && opt==1
          [theta,val]=minFunc(@drdae_discrim_joint_kl_obj_gpu_relu, theta, options, eI, ...
           data_cell, targets_cell, mixture_spectrum, false, false);
       elseif isGPU==1
          [theta,val]=minFunc(@drdae_discrim_joint_kl_obj_gpu, theta, options, eI, ...
            data_cell, targets_cell, mixture_spectrum, false, false);
       else
          [theta,val]=minFunc(@drdae_discrim_joint_kl_obj, theta, options, eI, ...
            data_cell, targets_cell, mixture_spectrum, false, false);
       end
    end

end

net.theta = theta;
net.eI = eI;
      

return;


%% unit test - small example
% context window size
context_win = 1;
% hidden units
hidden_units = 16;
num_layers = 1;
isdropout = 0;
% RNN temporal connection
isRNN = 2;
% One output source or two
iscleanonly = 0;
% Circular shift step
circular_step = 1000000;
% normalize input as L1 norm = 1
isinputL1 = 0;
% 0: MFCC, 1: logmel, 2: spectra
MFCCorlogMelorSpectrum = 2;
% feature frame rate
framerate = 64;

% discriminative training gamma parameter
pos_neg_r = 0.05;
% Last layer - linear or nonlinear
outputnonlinear = 0;
% soft mask obj
softabs = 1;
% 0: logistic, 1: tanh, 2: RELU
act = 2;
% constant for avoiding numerical problems
const = 1e-10;
% constant for avoiding numerical problems
const2 = 0.001;
% 0: not using GPU, 1: using GPU
isGPU = 0;

train_mode = 0;
% 0:'softlinear',1:'softabs', 2:'softquad', 3:'softabs_const',
% 4:'softabs_kl_const'
opt = 1;

[train1, fs, nbits]=wavread('female_train.wav');
[train2, fs, nbits]=wavread('male_train.wav');
maxLength=max([length(train1), length(train2)]);
train1(end+1:maxLength)=eps;
train2(end+1:maxLength)=eps;

max_iter = 30;

train_denoising({train1'}, {train2'}, context_win, hidden_units, num_layers, isdropout, ...
    isRNN, iscleanonly, circular_step , isinputL1, MFCCorlogMelorSpectrum, ...
    framerate, pos_neg_r, outputnonlinear, opt, act, train_mode, const,  ...
    const2, isGPU, max_iter)

%% unit test 2 - best setting:
% context window size
context_win = 3;
% hidden units
hidden_units = 1000;
num_layers = 3;
isdropout = 0;
% RNN temporal connection
isRNN = 2;
% One output source or two
iscleanonly = 0;
% Circular shift step
circular_step = 10000;
% normalize input as L1 norm = 1
isinputL1 = 0;
% 0: MFCC, 1: logmel, 2: spectra
MFCCorlogMelorSpectrum = 2;
% feature frame rate
framerate = 64;
% discriminative training gamma parameter
pos_neg_r = 0.05;
% Last layer - linear or nonlinear
outputnonlinear = 0;
% soft mask obj
softabs = 1;
% 0: logistic, 1: tanh, 2: RELU
act = 2;
% constant for avoiding numerical problems
const = 1e-10;
% constant for avoiding numerical problems
const2 = 0.001;
% 0: not using GPU, 1: using GPU
isGPU = 0;

train_mode = 0;
% 0:'softlinear',1:'softabs', 2:'softquad', 3:'softabs_const',
% 4:'softabs_kl_const'
opt = 1;

[train1, fs, nbits]=wavread('female_train.wav');
[train2, fs, nbits]=wavread('male_train.wav');
maxLength=max([length(train1), length(train2)]);
train1(end+1:maxLength)=eps;
train2(end+1:maxLength)=eps;

max_iter = 30;

train_denoising({train1'}, {train2'}, context_win, hidden_units, num_layers, isdropout, ...
    isRNN, iscleanonly, circular_step , isinputL1, MFCCorlogMelorSpectrum, ...
    framerate, pos_neg_r, outputnonlinear, opt, act, train_mode, const,  ...
    const2, isGPU, max_iter)
