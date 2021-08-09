%% sec2samp

function sample = sec2samp(second, sampling_freq)
sample = int64(sampling_freq*second + 1) ;
end