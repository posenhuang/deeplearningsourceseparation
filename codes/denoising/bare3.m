function bare3( x, y, m, n, w, lg, ti)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% Default bar width
if ~exist( 'w', 'var')
	w = .8;
end

% Deal with STOI scaling
up = 5*ceil(max( m(:))/5);
lo = 5*floor(min( n(:))/5);
y(end,:) = y(end,:)*up;
m(end,:) = m(end,:)*up;
n(end,:) = n(end,:)*up;

% Basic frame with gray grid
clf
axis( [0 5 lo up])
set( gca, 'ygrid', 'on', 'ycolor', [1 1 1]*.8, 'gridlinestyle', '-', 'yticklabel', {}, 'xtick', [])
a = ylim;

% Right axis for STOI
a1 = axes( 'position', get( gca, 'position'), 'xtick', [], ...
	'yaxislocation', 'right', 'color', 'none', 'ylim', a/up);
set( a1, 'ytick', linspace( 0, 1, 6));
ylabel( 'STOI')

% Left axis and plot with bells and whistles
axes( 'position', get( gca, 'position'), ...
	'color', 'none', 'ylim', a, 'xtick', 1:4);
hold on
h = bar( x, y, w);
hold off

% Add the min/max lines
for j = 1:size( y, 2)
	z = get( get( h(j), 'children'), 'xdata');
    if size(z, 1) > 0 
        for i = 1:size( y, 1)
            line( [z(1,i) z(1,i)]+(z(3,i)-z(1,i))/2, [m(i,j) n(i,j)], 'color', [.8 .8 .8], 'linewidth', 1.2);
            line( [z(1,i) z(1,i)]+(z(3,i)-z(1,i))/2, [m(i,j) n(i,j)], 'color', [.2 .2 .2]);
        end
    end
end

% Fluff
set( gca, 'xticklabel', {'SDR','SIR','SAR','STOI'})
xlabel( 'Metric')
ylabel( 'dB')
title( ti)
legend( lg{:}, 2, 'location', 'northwest')
%legend( lg{:}, 2, 'location', 'south', 'orientation', 'horizontal')
