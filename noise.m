function [outputimage] = noise(image,variance)
outputimage = image + uint8(normrnd(0,(variance*(1/255))^.5,size(image)));
end

