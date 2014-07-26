function htkwrite( DATA, Filename, HTKCode, sampPeriod )
% htkwrite( DATA, Filename, HTKCode, sampPeriod )
%
% Write DATA to HTK format file.
%
% Filename (string) - Name of the file to write to
% DATA (NFRAMES x NUMCOFS) - data to write
% HTKCode (scalar) - code describing write format
%  Default value: 512+64+3 means LPCC_C_E
% sampPeriod (scalar) - sample period, in 100ns units
%  Default value: 100000 means 10ms
%
% DATA must be already formatted for HTK, i.e. coefficients first, then
%  energy, then deltas if desired, then delta-energies if desired, etc.
%
% This function DOES NOT CHECK compatibility of HTKCode with DATA matrix.
%
% HTKCode is only checked to see whether or not we should perform compression.
% Compression is performed using the algorithm in 5.10 of the HTKBook.
% CRC is not supported by this function.
%
% Mark Hasegawa-Johnson
% July 3, 2002
% Based on function mfcc_write written by Alexis Bernard
%

% Find out whether or not compression is requested
if nargin<3, 
   HTKCode = 3; % Default code is LPCC 
   HTKCode = bitset(HTKCode, 7); % Set the energy bit
   HTKCode = bitset(HTKCode, 11); % Set the compression bit
end

% Open the file
fid=fopen(Filename,'w','ieee-be');
if fid<0,
    error(sprintf('Unable to write to file %s',Filename));
end

% Find nSamp and NCOFS
[ nSamp, NCOFS ] = size(DATA);

% Write sampPeriod
if nargin<4,
    sampPeriod = 100000;
end

% If data are compressed, write compression parameters and compressed data
if bitget(HTKCode, 11),
    sampSize = 2*NCOFS;
    %disp(sprintf('htkwrite: Writing %d frames, dim %d, compressed, to %s',nSamp,NCOFS,Filename));
    fwrite(fid,nSamp+4,'int32');
    fwrite(fid,sampPeriod,'int32');
    fwrite(fid,sampSize,'int16');
    fwrite(fid,HTKCode,'int16');

    % Create and write the compression parameters
    xmax = max(DATA);
    xmin = min(DATA);
    A = 2*32767./(xmax-xmin);
    B = (xmax+xmin)*32767 ./ (xmax-xmin);
    fwrite(fid, A, 'float');
    fwrite(fid, B, 'float');
    % Write compressed data
    Xshort = round( repmat(A,[nSamp 1]) .* DATA - repmat(B,[nSamp 1]) );
    fwrite(fid, Xshort', 'int16');   
else
    sampSize = 4*NCOFS;
    %disp(sprintf('htkwrite: Writing %d frames, dim %d, uncompressed, to %s',nSamp,NCOFS,Filename));
    fwrite(fid,nSamp,'int32');
    fwrite(fid,sampPeriod,'int32');
    fwrite(fid,sampSize,'int16');
    fwrite(fid,HTKCode,'int16');

    % Write uncompressed data
    fwrite(fid, DATA', 'float');
end

fclose(fid);