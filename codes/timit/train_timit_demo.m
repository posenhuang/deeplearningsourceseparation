function train_timit_demo(context_win, hidden_units, num_layers, isdropout, isRNN, iscleanonly,...
    circular_step , isinputL1, MFCCorlogMelorSpectrum, framerate, pos_neg_r, outputnonlinear, opt, act, train_mode, const, const2, isGPU)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
%%
rand('state',0)
randn('state',0)
%%
%% setup paths for code. assumed this script runs in its own directory
baseDir= '../../';
codeDir = [baseDir,'codes', filesep];
minFuncDir = [baseDir, 'tools', filesep, 'minFunc_2012', filesep];

saveDir = [codeDir,filesep,'timit',...
           filesep,'discrim_joint_offset_all_results'];

addpath([baseDir, filesep,'tools', filesep,'labrosa']);
addpath([baseDir, filesep,'tools', filesep,'bss_eval']);

addpath(baseDir);
addpath(genpath(minFuncDir));
addpath([baseDir,filesep,'codes',filesep,'timit', filesep,'Data_with_dev']);
addpath(codeDir);
addpath([codeDir,'timit']);

CFGPath=[baseDir,'tools',filesep,'htk_features', filesep];
addpath(CFGPath);

%% setup network architecture
eI = [];
% 0- mfcc, 1- logmel, 2- spectrum
eI.MFCCorlogMelorSpectrum=MFCCorlogMelorSpectrum; 
eI.CFGPath=CFGPath;
eI.seqLen = [1 10 25 50 100];
% eI.seqLen = [1];

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
% eI.seqLen = [1 10 25 50 100];
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
    modelname=['model_RNN',num2str(isRNN)];
else
    modelname='model_DNN';
end
modelname=[modelname,'_win',num2str(context_win),'_h',num2str(hidden_units),'_l',num2str(num_layers)];
if iscleanonly, modelname=[modelname,'_cleanonly']; end
if isdropout,  modelname=[modelname,'_dropout'];    end
modelname=[modelname,['_r', num2str(eI.r)]];

modelname=[modelname,['_', num2str(eI.framerate),'ms']];
% modelname=[modelname,'_off',num2str(offset_step), '_snr', num2str(SNR_step)];
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
options.MaxIter = 800;
options.MaxFunEvals = 2500;
options.Corr = 50;
options.DerivativeCheck = 'off';
options.outputFcn = @save_callback_timit_general;

%% compute feature
SNRs=0;

[train1, fs, nbits]=wavread('female_train.wav');
[train2, fs, nbits]=wavread('male_train.wav');

maxLength=max([length(train1), length(train2)]);
train1(end+1:maxLength)=eps;
train2(end+1:maxLength)=eps;

train1=train1./sqrt(sum(train1.^2));
train2=train2./sqrt(sum(train2.^2));

eI.fs=fs;
%%
% chunk
[data_cell, targets_cell, mixture_spectrum]=formulate_data(train1, train2, eI, eI.train_mode); %0 -- chunk, 2--no chunk

  global SDR;
  SDR.deviter=0;   SDR.devmax=0;   SDR.testmax=0;

  eI.writewav=0;

if isGPU==1
  [theta,val]=minFunc(@drdae_discrim_joint_kl_obj_gpu, theta, options, eI, data_cell, targets_cell, mixture_spectrum, false, false);
else
  [theta,val]=minFunc(@drdae_discrim_joint_kl_obj, theta, options, eI, data_cell, targets_cell, mixture_spectrum, false, false);
end

  fprintf('%s\tdevmaxiter:\t%d\tdevSDR:\t%.3f\ttestSDR:\t%.3f\n',modelname, SDR.deviter, SDR.devmax, SDR.testmax);

return;

%% unit test

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

train_timit_demo(context_win, hidden_units, num_layers, isdropout, isRNN, iscleanonly,...
    circular_step , isinputL1, MFCCorlogMelorSpectrum, framerate, pos_neg_r, ...
    outputnonlinear, opt, act, train_mode, const, const2, isGPU)


