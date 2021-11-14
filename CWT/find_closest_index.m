function [start_idx,finish_idx] = find_closest_index(array, start_val, finish_val)
temp_arr = abs(array - start_val*ones(size(array)));
possible_min = min(temp_arr);
start_idx = find(temp_arr == possible_min);
temp_arr = abs(array - finish_val*ones(size(array)));
possible_max = min(temp_arr);
finish_idx = find(temp_arr == possible_max);
end