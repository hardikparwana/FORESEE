% Paper Title: A Generalized Unscented Transformation for Probability Distributions
%
function [mu, second_cen_moment, third_cen_moment, ...
    fourth_cen_moment, Ex] =  Beta_moments(alpha, beeta)
% INPUTS
%
%   alpha                       -   first shape parameter
%   beeta                        -   second shape parameter
%
% OUTPUTS
%
%   mu                      -   Mean of random variable
%   second_cen_moment       -   Variance of random variable
%   third_cen_moment        -   Skewness of random variable
%   fourth_cen_moment       -   Kurtosis of random variable
%   Ex                      -   Vector of first four raw mments


% syms alpha beta

Ex1 = (alpha)/(alpha + beeta);                   % E [ x ]
Ex2 = (alpha + 1)/(alpha + beeta + 1)*Ex1;       % E [ x^2 ]
Ex3 = (alpha + 2)/(alpha + beeta + 2)*Ex2;       % E [ x^3 ]
Ex4 = (alpha + 3)/(alpha + beeta + 3)*Ex3;       % E [ x^4 ]

% The mean
mu = Ex1;     
% Variance  --->   E[ (x - mu)^2 ]
second_cen_moment = (Ex2 - mu*mu);   
% Skew      --->   E[ (x - mu)^3 ]
third_cen_moment = (Ex3 - 3*Ex2*mu + 3*mu*mu^2 - mu^3);  
% Kurtosis  --->   E[ (x - mu)^4 ]
fourth_cen_moment = (Ex4 - 4*Ex3*mu + 6*Ex2*mu^2 - 4*mu*mu^3 + mu^4);  

%% Save output values as double
Ex1 = double(Ex1);      Ex2 = double(Ex2);
Ex3 = double(Ex3);      Ex4 = double(Ex4);

Ex = [Ex1; Ex2; Ex3; Ex4];
mu = double(mu);             
second_cen_moment = double(second_cen_moment);
third_cen_moment = double(third_cen_moment);
fourth_cen_moment = double(fourth_cen_moment);

end