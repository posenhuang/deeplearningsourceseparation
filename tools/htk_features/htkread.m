function [ DATA, HTKCode ] = htkread( Filename )
% [ DATA, HTKCode ] = htkread( Filename )
%
% Read DATA from possibly compressed HTK format file.
%
% Filename (string) - Name of the file to read from
% DATA (nSamp x NUMCOFS) - Output data array
% HTKCode - HTKCode describing file contents
%
% Compression is handled using the algorithm in 5.10 of the HTKBook.
% CRC is not implemented.
%
% Mark Hasegawa-Johnson
% July 3, 2002
% Based on function mfcc_read written by Alexis Bernard
%

fid=fopen(Filename,'r','b');
if fid<0,
    error(sprintf('Unable to read from file %s',Filename));
end

% Read number of frames
nSamp = fread(fid,1,'int32');

% Read sampPeriod
sampPeriod = fread(fid,1,'int32');

% Read sampSize
sampSize = fread(fid,1,'int16');

% Read HTK Code
HTKCode = fread(fid,1,'int16');

%%%%%%%%%%%%%%%%%
% Read the data
if bitget(HTKCode, 11),
    DIM=sampSize/2;
    nSamp = nSamp-4;
%    disp(sprintf('htkread: Reading %d frames, dim %d, compressed, from %s',nSamp,DIM,Filename)); 

    % Read the compression parameters
    A = fread(fid,[1 DIM],'float');
    B = fread(fid,[1 DIM],'float');
    
    % Read and uncompress the data
    DATA = fread(fid, [DIM nSamp], 'int16')';
    DATA = (repmat(B, [nSamp 1]) + DATA) ./ repmat(A, [nSamp 1]);

    
else
    DIM=sampSize/4;
%    disp(sprintf('htkread: Reading %d frames, dim %d, uncompressed, from %s',nSamp,DIM,Filename)); 

    % If not compressed: Read floating point data
    DATA = fread(fid, [DIM nSamp], 'float')';
end

fclose(fid);
