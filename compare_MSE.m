function output = compare_MSE(edgepercent,cornerpercent,pluspercentmajor,pluspercentminor,img,ogimg)
    imgsize = size(img);
    pixels = imgsize(1)*imgsize(2);
    edgePImg = dftfilter(edgePass(edgepercent,edgepercent,imgsize),img);
    cornerPImg = dftfilter(cornerPass(cornerpercent,imgsize),img);
    plusBImg = dftfilter(plusBlock(pluspercentmajor,pluspercentminor,imgsize),img);
    noneMSE = mse(img,ogimg);
    edgePMSE = mse(edgePImg,ogimg);
    cornerPMSE = mse(cornerPImg,ogimg);
    plusBMSE = mse(plusBImg,ogimg);
    output = [noneMSE,edgePMSE,cornerPMSE,plusBMSE];
end

