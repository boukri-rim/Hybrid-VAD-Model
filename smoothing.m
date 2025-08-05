function [y_net]= smoothing(X,y_net)
%(length(X)- length(test_data))
%////////////le lissage
y_net(length(X)+1)=0;
y_net(length(X)+2)=0;
for i=3:length(X)
%for i=3:(length(X)- length(test_data))
     if (y_net(i)==0 && y_net(i-1)==1 && y_net(i+1)==1)
         y_net(i)=y_net(i-1);
     else
         y_net(i)=y_net(i);
     end
      if (y_net(i)==1 && y_net(i-1)==0 && y_net(i+1)==0)
         y_net(i)=y_net(i-1);
     else
         y_net(i)=y_net(i);
      end
    if (y_net(i-1)==1 && y_net(i)==0 && y_net(i+1)==0 && y_net(i+2)==1) 
        y_net(i)=1;
        y_net(i+1)=1;
    else
        y_net(i)=y_net(i);
        y_net(i+1)= y_net(i+1);
    end
    if (y_net(i-1)==0 && y_net(i)==1 && y_net(i+1)==1 && y_net(i+2)==0) 
        y_net(i)=0;
        y_net(i+1)=0;
    else
        y_net(i)=y_net(i);
        y_net(i+1)=y_net(i+1);
    end       
end
y_net=y_net(1:length(X));
end