function filter =  plusBlock(edgepercent,cornerpercent, imgsize)
    filter = min(edgePass(edgepercent,cornerpercent, imgsize),edgePass(cornerpercent,edgepercent, imgsize));
end
