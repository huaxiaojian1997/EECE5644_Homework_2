% For case1: Number of samples = 400; class means [0,0]T and [3,3]T ; class covariance matrices both set to I; equal class priors.
clear all, close all,
syms correct error  P_error,
n = 2; % number of feature dimensions
N = 400; % number of iid samples
mu(:,1) = [0;0]'; mu(:,2) = [3;3]';
Sigma(:,:,1) = eye(2); Sigma(:,:,2) = eye(2);
p = [0.5,0.5]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    %x(:,label==l) = randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1));
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

subplot(4,1,1);
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive

correct = length(ind00) + length(ind11); % the number of correct
error = length(ind10) + length(ind01); % the number of error
P_error = error/N; % probability of error, empirically estimated
fprintf('the number of correct is %d',correct);
fprintf('\n');
fprintf('the number of error is %d',error);
fprintf('\n');
fprintf('the probability of error is %.4f',P_error);

subplot(4,1,2);% class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,axis equal,
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1'), 
title('Data and their inferred (decision) labels'),
xlabel('x_1'), ylabel('x_2'), 

% Appending LDA to the ERM code for TakeHomeQ3...
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly
subplot(4,1,3);
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), axis equal,
legend('Class 0','Class 1'), 
title('LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 
tau = 0;
decisionLDA = (yLDA >= 0);

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(yLDA,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(yLDA,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));
Ind00 = find(decision==0 & label==0); % probability of true negative
Ind10 = find(decision==1 & label==0); % probability of false positive
Ind01 = find(decision==0 & label==1); % probability of false negative
Ind11 = find(decision==1 & label==1); % probability of true positive
subplot(4,1,4);% class 0 circle, class 1 +, correct green, incorrect red
plot(yLDA(1,ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(1,ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(1,ind01),zeros(1,length(ind01)),'+r'); hold on,
plot(yLDA(1,ind11),zeros(1,length(ind11)),'+g'); hold on,axis equal,
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1'), 
title('LDA projection of Data and their inferred (decision) labels'),
xlabel('x_1'), ylabel('x_2'), 