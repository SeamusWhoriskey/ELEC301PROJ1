function spectrum = dftspectrum(img)
    complexSpectrum = fft(img);
    spectrum = abs(complexSpectrum);
    spectrum(:,:,2) = angle(complexSpectrum);
end

