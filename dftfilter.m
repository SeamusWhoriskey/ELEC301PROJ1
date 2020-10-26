function output = dftfilter(filter,img)
    spectrum = fft2(img);
    newspec = spectrum.*filter;
    output = uint8(ifft2(newspec,'symmetric'));
end

