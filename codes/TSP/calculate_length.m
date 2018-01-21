% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
fs=16000;

len =[];
for s1id = {'FA', 'MC', 'FB', 'MD'}
    filenames=dir(['Data',filesep, s1id{1}, filesep,'*.wav']);
    % load 60 wav files at the same time (source 1)
    s1(:,1)=cellfun(@(x) audioread(['Data', filesep, s1id{1}, filesep, x]), {filenames.name}, 'UniformOutput', 0);
    
    
    % downsample to 16kHz
%     s1=cellfun(@(x) resample(x, fs, 48000), s1, 'UniformOutput', 0);

    for i = 1:size(s1,1)
       len = [len, size(s1{i},1)/48000]; % in second
    end
end

mean(len);

