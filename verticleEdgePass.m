function filter =  verticleEdgePass(percent, imgsize)
    filter = ones(imgsize);
    middle = zeros(uint16(imgsize.*[percent,1]));
    middlesize = size(middle);
    edge1 = uint16((imgsize(1)-middlesize(1))/2);
    edge2 = edge1+middlesize(1)-1;
    filter(edge1:edge2,:) = middle;
end

