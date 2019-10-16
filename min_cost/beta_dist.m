% beta distribution 
close all
X = 0:.01:1;
y1 = betapdf(X,50, 100);


figure
plot(X,y1,'Color','r','LineWidth',2)


