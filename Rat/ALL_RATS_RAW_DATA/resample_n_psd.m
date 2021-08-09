for animal = [15,16,17,86,87,88,89,90,91,92,103,104]

    CH1 = load(['Voltage_Animal',num2str(animal),'CH1.mat']);
    CH1 = CH1.Voltage_CH1;
    CH2 = load(['Voltage_Animal',num2str(animal),'CH2.mat']);
    CH2 = CH2.Voltage_CH2;
    Upsampling_Rate = 1;
    Downsampling_Rate = 2;
    F_s = 1000;
    Taper_number = 15;

    RS_CH1 = resample(CH1',Upsampling_Rate,Downsampling_Rate)';
    pxx = pmtm(RS_CH1',Taper_number,size(RS_CH1,2),F_s*Upsampling_Rate/Downsampling_Rate)';
    PSD_CH1 = pxx;
    
    RS_CH2 = resample(CH2',Upsampling_Rate,Downsampling_Rate)';
    pxx = pmtm(RS_CH2',Taper_number,size(RS_CH2,2),F_s*Upsampling_Rate/Downsampling_Rate)';
    PSD_CH2 = pxx;   
    

    save(['PSD_ANIMAL',num2str(animal),'_CH1.mat'],'PSD_CH1')
    save(['PSD_ANIMAL',num2str(animal),'_CH2.mat'],'PSD_CH2')
end