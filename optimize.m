function  optimize(ogimgs,imgs,vars,subset,points)
    percents = linspace(0,.99,points);
    numimgs = length(ogimgs);
    numvars = length(vars);
    normalized_congregate_mse = zeros([points,3]);
    for i = 1:numimgs
       ogimg = ogimgs{i};
       varimgs = imgs{i};
       for j = subset
           img = varimgs{j};
           for k = 1:points
              mses = compare_MSE(percents(k),percents(k),percents(k),.997,img,ogimg);
              mses = mses./mses(1);
              normalized_congregate_mse(k,:) = normalized_congregate_mse(k,:)+mses(2:4);
           end
       end
    end
    normalized_congregate_mse = normalized_congregate_mse./(numimgs*numvars);
    plot(percents,normalized_congregate_mse')
    legend('edgePass','cornerPass','plusBlock')
end

