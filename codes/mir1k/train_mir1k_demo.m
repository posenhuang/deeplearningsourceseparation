function train_mir1k_demo(context_win, hidden_units, num_layers, isdropout, ...
    isRNN, iscleanonly, circular_step , isinputL1, MFCCorlogMelorSpectrum, ...
    framerate, pos_neg_r, outputnonlinear, opt, act, train_mode, const,  ...
    const2, isGPU)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% Demo MIR-1K training ---------------------------------------------
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

rand('state',0)
randn('state',0)

%% setup paths for code. assumed this script runs in its own directory
% CHANGE baseDir to the top of code directory
baseDir= '../../';
codeDir = [baseDir,'codes', filesep];
minFuncDir = [baseDir, 'tools', filesep, 'minFunc_2012', filesep];

saveDir = [codeDir,filesep,'mir1k',filesep,'discrim_joint_offset_results'];

%% add paths
addpath([baseDir, filesep,'tools', filesep,'labrosa']);
addpath([baseDir, filesep,'tools', filesep,'bss_eval']);
addpath([baseDir, filesep,'tools', filesep,'bss_eval_3']);
addpath(baseDir);
addpath(genpath(minFuncDir));
addpath(codeDir);
addpath([codeDir,'mir1k']);

CFGPath=[baseDir,'tools',filesep,'htk_features', filesep];
addpath(CFGPath);

%% setup network architecture
eI = [];
% 0- mfcc, 1- logmel, 2- spectrum
eI.MFCCorlogMelorSpectrum=MFCCorlogMelorSpectrum;
eI.CFGPath=CFGPath;
eI.seqLen = [1 10 25 50 100];

eI.framerate=framerate;
if eI.framerate==64
    eI.winsize = 1024;    eI.nFFT = 1024;    eI.hop =eI.winsize/2;    eI.scf=1;%scf = 2/3;
else %32
    eI.winsize = 512;    eI.nFFT = 512;    eI.hop = eI.winsize/2;    eI.scf=1;%scf = 2/3;
end
winsize=eI.winsize; nFFT= eI.nFFT; hop= eI.hop; scf=eI.scf;
windows=sin(0:pi/winsize:pi-pi/winsize);

% single target or multiple targets
eI.cleanonly=iscleanonly;

% context window size of the input.
eI.num_contextwin = context_win;

% dimension of each input frame
if eI.MFCCorlogMelorSpectrum==0 %0 for mfcc 1 for logmel
    eI.featDim =39;
elseif eI.MFCCorlogMelorSpectrum==1 %0 for mfcc 1 for logmel
    eI.featDim =123;
else
    eI.featDim = (eI.nFFT/2+1);
end

eI.dropout = isdropout;

% weight tying in hidden layers
% if you want tied weights, must have odd number of *hidden* layers
eI.tieWeights = 0;

eI.const=const; eI.const2=const2;

hidden_units_set=[];

for il=1:num_layers
    hidden_units_set=[hidden_units_set, hidden_units];
end
% 2 hidden layers and output layer
if eI.cleanonly==1,
    eI.layerSizes = [hidden_units_set  nFFT/2+1 ];
else
    eI.layerSizes = [hidden_units_set  (nFFT/2+1)*2];
end
% highest hidden layer is temporal
eI.temporalLayer =isRNN;
% dim of network input at each timestep (final size after window & whiten)
eI.inputDim = eI.featDim * eI.num_contextwin;
% length of input sequence chunks.
% activation function
switch act
    case 0
        eI.activationFn = 'logistic';
    case 1
        eI.activationFn = 'tanh';
    case 2
        eI.activationFn = 'RELU';
end

% temporal initialization type
eI.temporalInit = 'rand';
% weight norm penaly
eI.lambda = 0;
% file containing whitening matrices for outputs
eI.outputL1=0;

eI.inputL1=isinputL1;

eI.r=pos_neg_r;

eI.isdiscrim=2;

if opt==0,
    eI.opt='softlinear';
elseif opt==1,
    eI.opt='softabs';
elseif opt==2,
    eI.opt='softquad';
elseif opt==3,
    eI.opt='softabs_const';
elseif opt==4,
    eI.opt='softabs_kl_const';
end

eI.train_mode= train_mode;
eI.outputnonlinear=outputnonlinear;

%% setup weight caching
if isRNN,
    if isRNN==num_layers+1
        modelname=['model_RNNall'];
        eI.fullRNN=1;
    else
        modelname=['model_RNN',num2str(isRNN)];
    end
else
    modelname='model_DNN';
end
modelname=[modelname,'_win',num2str(context_win),'_h', num2str(hidden_units),'_l',num2str(num_layers)];
if iscleanonly, modelname=[modelname,'_cleanonly']; end
if isdropout,  modelname=[modelname,'_dropout',num2str(isdropout)];    end
modelname=[modelname,['_r', num2str(eI.r)]];

modelname=[modelname,['_', num2str(eI.framerate),'ms']];
modelname=[modelname, '_', num2str(circular_step)];
eI.circular_step = circular_step;

modelname=[modelname,'_',eI.opt];
if outputnonlinear==0, modelname=[modelname,'_linearout']; end

modelname=[modelname, '_', eI.activationFn];

if eI.inputL1, modelname=[modelname, '_L',num2str(eI.inputL1)]; end

if eI.MFCCorlogMelorSpectrum==0
    modelname=[modelname,'_mfcc'];
elseif  eI.MFCCorlogMelorSpectrum==1
    modelname=[modelname,'_logmel'];
elseif  eI.MFCCorlogMelorSpectrum==2
    modelname=[modelname,'_spectrum'];
else
    modelname=[modelname,'_logpowspect'];
end

modelname= [modelname, '_trn', num2str(eI.train_mode)];
modelname=[modelname,'_c',num2str(const), '_c',num2str(const2)];

eI.modelname=modelname;
disp(modelname);

eI.saveDir = [saveDir, filesep, modelname, filesep];
if ~exist(eI.saveDir,'dir'), mkdir(eI.saveDir); end

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
options.MaxIter = 400;
% options.MaxIter = 50;
options.MaxFunEvals = 2500;
options.Corr = 50;
options.DerivativeCheck = 'off';
% options.DerivativeCheck = 'on';
options.outputFcn = @save_callback_mir1k_general;

eI.DataPath=[codeDir,'mir1k', filesep, 'Wavfile',filesep];

train_files= dir( [eI.DataPath, 'train',filesep,'*wav']);

% chunk
[data_cell, targets_cell, mixture_spectrum]=formulate_data(train_files, eI, eI.train_mode); %0 -- chunk, 2--no chunk, 3- icassp

%% BSS EVAL setting
global SDR;
SDR.deviter=0;   SDR.devmax=0;   SDR.testmax=0;

global SDR_bss3;
SDR.devsar=0; SDR.devsir=0; SDR.testsar=0; SDR.testsir=0;
SDR_bss3.deviter=0;   SDR_bss3.devmax=0;   SDR_bss3.testmax=0;
SDR_bss3.devsar=0; SDR_bss3.devsir=0; SDR_bss3.testsar=0; SDR_bss3.testsir=0;
eI.bss3=1;

eI.writewav=0;

%% run optimizer
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
% % for non joint training
% eI.isdiscrim=1;
% [theta,val]=minFunc(@drdae_discrim_obj, theta, options, eI, data_cell, ...
%                     targets_cell, false, false);


fprintf('%s\tdevmaxiter:\t%d\tdevGNSDR:\t%.3f\ttestGNSDR:\t%.3f\t',...
    eI.modelname, SDR.deviter, SDR.devmax, SDR.testmax);
fprintf('devGSIR:\t%.3f\tdevGSAR:\t%.3f\t',SDR.devsir, SDR.devsar);
fprintf('testGSIR:\t%.3f\ttestGSAR:\t%.3f\n',SDR.testsir, SDR.testsar);

fprintf('%s\tbss3 devmaxiter:\t%d\tdevGNSDR:\t%.3f\ttestGNSDR:\t%.3f\t', ...
    eI.modelname, SDR_bss3.deviter, SDR_bss3.devmax, SDR_bss3.testmax);
fprintf('devGSIR:\t%.3f\tdevGSAR:\t%.3f\t',SDR_bss3.devsir, SDR_bss3.devsar);
fprintf('testGSIR:\t%.3f\ttestGSAR:\t%.3f\n',SDR_bss3.testsir, SDR_bss3.testsar);

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

train_mir1k_demo(context_win, hidden_units, num_layers, isdropout, ...
    isRNN, iscleanonly, circular_step , isinputL1, MFCCorlogMelorSpectrum, ...
    framerate, pos_neg_r, outputnonlinear, opt, act, train_mode, const,  ...
    const2, isGPU)

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

train_mir1k_demo(context_win, hidden_units, num_layers, isdropout, ...
    isRNN, iscleanonly, circular_step , isinputL1, MFCCorlogMelorSpectrum, ...
    framerate, pos_neg_r, outputnonlinear, opt, act, train_mode, const,  ...
    const2, isGPU)
