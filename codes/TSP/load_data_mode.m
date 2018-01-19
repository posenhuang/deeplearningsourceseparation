function [mixture, signal_1, signal_2]=load_data_mode(speaker_setting, data_mode, data_path)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
%% load signals and STFT
% eI -- use eI.data_mode
% mode -- 0: training, 1: valid, 2: testing

% pick up the TSP speakers: TSP speaker IDs used
% FA vs FB (female vs female)
% MC vs MD (male vs male)
% FA vs MC (male vs female)
 
if speaker_setting==0, 
    s1id='FA'; s2id='MC';
elseif speaker_setting==1, 
    s1id='FA'; s2id='FB';
elseif speaker_setting==2,
    s1id='MC'; s2id='MD';
else
   fprintf('invalid data_mode');
end
  
fs=16000;
 
filenames=dir([data_path, filesep, s1id, filesep,'*.wav']);
% load 60 wav files at the same time (source 1)
s1(:,1)=cellfun(@(x) audioread([data_path, filesep, s1id, filesep, x]), {filenames.name}, 'UniformOutput', 0);
% downsample to 16kHz
s1=cellfun(@(x) resample(x, fs, 48000), s1, 'UniformOutput', 0);
% group signals
train1 = cell2mat(s1);
test1 = train1(round(.9*length(train1)):end);
train1(round(.9*length(train1)):end)=[];
val1 = train1(round(8/9*length(train1)):end);
train1(round(8/9*length(train1)):end)=[];
 
% ditto for source 2

filenames=dir([data_path,filesep, s2id, filesep,'*.wav']);
s2(:,1)=cellfun(@(x) audioread([data_path, filesep, s2id, filesep, x]), {filenames.name}, 'UniformOutput', 0);
s2=cellfun(@(x) resample(x, fs, 48000), s2, 'UniformOutput', 0);
train2 = cell2mat(s2);
test2 = train2(round(.9*length(train2)):end);
train2(round(.9*length(train2)):end)=[];
val2 = train2(round(8/9*length(train2)):end);
train2(round(8/9*length(train2)):end)=[];
 
% normalize (0 dB SNR)
minLength=min([length(train1), length(train2)]);
train1(minLength+1:end)=[];
train2(minLength+1:end)=[];
train1=train1./sqrt(sum(train1.^2));
train2=train2./sqrt(sum(train2.^2));
 
minLength=min([length(test1), length(test2)]);
test1(minLength+1:end)=[];
test2(minLength+1:end)=[];
test1=test1./sqrt(sum(test1.^2));
test2=test2./sqrt(sum(test2.^2));
 
minLength=min([length(val1), length(val2)]);
val1(minLength+1:end)=[];
val2(minLength+1:end)=[];
val1=val1./sqrt(sum(val1.^2));
val2=val2./sqrt(sum(val2.^2));
 
 
% mixing
trainx=train1+train2;
valx=val1+val2;
testx=test1+test2;

if data_mode==0,
    mixture=trainx; signal_1=train1; signal_2= train2;
elseif data_mode==1,
    mixture=valx; signal_1=val1; signal_2=val2;
elseif data_mode==2,
    mixture=testx; signal_1=test1; signal_2=test2;
else
    fprintf('incorrect data mode');
end


