function [ts,sr]=readnist(filename,sr);

% READNIST
%
%  [ts,sr]= READNIST (filename,[sr])
%
%  Reads NIST WAV/SPHERE files, such as those in N/C/TIMIT or
%  Switchboard or Mandarin-VOA TDT files.
%
%  filename can be provided with or without the .wav/.sph/.WAV/.SPH extension.
%
%  Example usage: to read '/blah/blah1/timit/dr1/fcjf0/sa1.wav' just say
%    [ts,sr] = readnist ('/blah/blah1/timit/dr1/fcjf0/sa1');
%  or 
%    [ts,sr] = readnist ('/blah/blah1/timit/dr1/fcjf0/sa1.wav');
% 
%  To read the ntimit file '/blah/blah1/ntimit/dr1/fcjf0/SA1.SPH'
%  just say
%    [ts,sr] = readnist ('/blah/blah1/ntimit/dr1/fcjf0/SA1');
%  or
%    [ts,sr] = readnist ('/blah/blah1/ntimit/dr1/fcjf0/SA1.SPH');
%  
%  ts is a Nx1 vector of entries representing the sound sample in filename
%  sr represents the sample rate, and you DON'T have to supply it - this
%     function works it out from the file headers automatically.
%     If you do supply it, any sample rate info contained in headers is ignored.
%  The sr that is returned is 
%    : the one input, or (if none is input)
%    : the sample rate in the file, or 
%    : 0 (if no sample rate is supplied by the user or the file headers)
%
%  The first line of filename should have NIST_1A in its header
%    otherwise ts = [] and sr = 0
%
% Assumptions:
%  - first line of file has NIST_1A
%  - second line has an integer, usually 512 or 1024, that has the number of
%     bytes in the header
%  - the header has a line of the form "sample_rate -i D", where D is an 
%     integer like 16000 that is the sampling rate in Hertz. 
%  - the header has a line with "sample_byte_format" (if not, little endian
%     is assumed) and if so, that this line is either 
%     "sample_byte_format -s2 01" (little endian) or
%     "sample_byte_format -s2 10" (big endian) 
%
% You should be able to play the sound with the Matlab command
%   soundsc (ts,sr)
% afterwards, assuming your sound card exists and works. (If it doesnt, 
% pray hard, and try again.)
%
% This file has only been tested on TIMIT, NTIMIT, CTIMIT and Switchboard (ICSI-WS97) files.
%
%	Author : D.Surendran 04/24/04 (dinojatcsdotuchicagodotedu)
%
% NOTE ON SHORTENED FILES 
% Update: if you are on a unix-based system, you can also read
%   'shorten'ed files. This requires that you change the current
%   Matlab directory to one where you have write permission and
%   enough space to store an uncompressed version of the sound
%   file. The uncompressed file will be deleted afterwards, dont
%   worry! 
%     The current directory should also have a copy of shorten 
%   (If not, modify the command in this file that reads
%
%     cmd = ['shorten -x -d' num2str(headerlen) ' ' filename ' ' tmpfilename ';'];
%
 
% Acknowledgements to documentation http://ftp.cwi.nl/audio/NIST-SPHERE  
% and the readtimit.m file by C.E.Ho 8/20/97 (which this file is meant to replace)


ts = [];

if nargin < 2
  sr = 0;
end
if nargin < 1
  error ('You need to have at least one input!');
end

exts = {'wav','WAV','sph','SPH'};
extsor = '';
if ~exist (filename)
  for i = 1 : length (exts)
    extsor = [extsor '.' exts{i}];
    if (i ~= length(exts))
      extsor = [extsor ' or '];
    end
    fn = [filename '.' exts{i}];
    if exist (fn)
      filename = fn;
      break;
    end
  end
end
if ~exist (filename)
  error (sprintf ('%s not found (even with %s added)',filename, extsor));
end   

s=textread(filename,'%s',2);

if length(s) >= 1
  a= findstr(s{1},'NIST_1A');
  if ~(length(a)) | ~a
    error (sprintf('Not a NIST Sphere file : first line of %s doesnt contain NIST_1A',filename));
  end
else
  error (sprintf('Not a NIST Sphere file : first line of %s isnt a header',filename));
end

if length(s) >= 2
  headerlen = floor (str2num (s{2}));
  if 0 == headerlen
    error (sprintf('Second line of %s should have the header length in bytes',filename));
  end
end

fid = fopen (filename,'rt','ieee-le');             % doesnt matter which endian it is when reading ascii headers
[a,count] = fread (fid,headerlen,'uchar');         % uchar is default anyway
header = char(a');                                 % a is a vector of ascii values,
                                                   % header is an ascii string
						   
headersplit = strread (header, '%s', length(header),'delimiter','\n');

% headersplit{i} has the i-th line of the file header

endian = 0;
useshorten = 0;
for i = 1 : length(headersplit)
  if findstr (headersplit{i},'sample_rate') & (0 == sr)
    sr = strread (headersplit{i}, 'sample_rate -i %d');
  end
  if findstr (headersplit{i},'sample_byte_format')
    endian = strread (headersplit{i}, 'sample_byte_format -s2 %d');
  end
  if findstr (headersplit{i},'sample_coding') & findstr (headersplit{i},'shorten')
    useshorten = 1;
  end
end

if (0 == sr)
  error ('What is the sample rate for this file? Cant find the sample_rate line');
end
if (0 == endian)
  warning (sprintf ('%s does not have byte format coded, assuming little endian'));
  endian = 'le';
elseif (1 == endian)
    endian = 'le';
elseif (10 == endian)
  endian = 'be';
else
  error ('error parsing line with sample_byte_format - cant work out whether this file is little or big endian');
end    
fclose (fid);

if 1 == useshorten 
  if ~isunix
     error ([filename 'needs to be uncompressed with Tony Robinsons' ...
	     'program - this can be done, but only when' ...
	     'this Matlab code is run on a unix-based system.' ...
	     'I cant get the ! command to behave like the' ...
	     'unix command in Matlab. Sorry! Note that' ...
	     'readnist works ok on windows for unshortened' ...
	     'NIST_1A sound files.']);
  end     
  if 0 == exist ('shorten')
     error ([filename 'needs to be uncompressed with Tony Robinsons' ...
	     ' shorten program. Please installit from' ...
	     ' http://www.hornig.net/shorten/ and add its'...
	     ' directory to the Matlab path.']);
  end		      
  
  tmpfilename = ['uncompressed_' filename];

  cmd = ['./shorten -x -d' num2str(headerlen) ' ' filename ' ' tmpfilename ';'];
  unix (cmd);

  fid=fopen(tmpfilename,'r',['ieee-' endian] );
else
  fid=fopen(filename,'r',['ieee-' endian] );
end;
  
junk=fread(fid,headerlen,'short');
ts=fread(fid,Inf, 'short');
fclose(fid);

if (1 == useshorten) & isunix
  unix (['rm ' tmpfilename]);
else
  warning (['temporary file ' tmpfilename ' still exists!']);
end  
	   
	    




