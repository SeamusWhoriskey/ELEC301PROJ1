function filter =  cornerPass(percent, imgsize)
    filter = min(verticleEdgePass(percent, imgsize),verticleEdgePass(percent, flip(imgsize))');
end

