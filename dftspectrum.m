function spectrum = dftspectrum(img)
    complexSpectrum = fft2(img);
    spectrum = abs(complexSpectrum);
    spectrum(:,:,2) = angle(complexSpectrum);
end

