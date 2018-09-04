function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

pos_index = find(y==1);
neg_index = find(y==0); 
plot(X(pos_index,1),X(pos_index,2),'k+','LineWidth',2,'MarkerSize',7);
plot(X(neg_index,1),X(neg_index,2),'ko','MarkerFaceColor','y','MarkerSize',7);

% pos_data = X(find(y),:);
% neg_data = X(find(~y),:);
% plot(pos_data(:,1),pos_data(:,2),'k+',neg_data(:,1),neg_data(:,2),'ko');








% =========================================================================



hold off;

end
