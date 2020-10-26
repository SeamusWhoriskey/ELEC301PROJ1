function filter =  edgePass(percent1,percent2, imgsize)
    filter = ones(imgsize);
    middle = zeros(uint16(imgsize.*[percent1,percent2]));
    middlesize = size(middle);
    edge11 = uint16((imgsize(1)-middlesize(1))/2);
    edge12 = edge11+middlesize(1)-1;
    edge21 = uint16((imgsize(2)-middlesize(2))/2);
    edge22 = edge21+middlesize(2)-1;
    filter(edge11:edge12,edge21:edge22) = middle;
    %filter = complex(filter,ones(size(filter)));
end
