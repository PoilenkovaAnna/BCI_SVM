
MIN_PHONEME_ID = 1

def extract_strict_sectors(edf, sector_length = 600 ):
	"""
	Extract sectors of the given length using labels.
	Sample usage is extracting sectors of length 600 ms (1000Hz).

	Returns segments[begin,end] and labels

		** 1n perception / n pronunciation
	"""
	sectors = []
	labels = []

	number_of_current_phoneme = None
	counter = 0
	silent_speach = False

	label_channel = edf['METKA']
	X = label_channel[1]
	Y = label_channel[0].T[:,0]

	for index, (timestamp, value) in enumerate(zip(X, Y)):
		counter-=1
		if value > 0:
			value = int(value)

			# segment begin of silent speach
			if value // 10 == 0:
				counter = sector_length + 100
				number_of_current_phoneme = value % 10
				silent_speach = True
			else:
				silent_speach = False # another label

		if silent_speach and counter == 0:
			sectors.append((index - sector_length, index))
			labels.append(number_of_current_phoneme)

	return sectors, labels


def subselect_channels(edf, target_channels):
    all_channels = edf.ch_names
    channels = []

    for target_channel in target_channels:
        for channel_name_optiont in target_channel:
            if channel_name_optiont in all_channels:
                data_channel = edf[channel_name_optiont]
                Y_channel = list(data_channel[0][0]) # Значения сигнала
                X_channel = data_channel[1] # Время
                channels.append(Y_channel)

    if len(channels) != len(target_channels):
         raise RuntimeError(f'Not all channels found')
    else:
        return channels

def split_sectors(channels_data, sectors):
    '''
    Select segments from data by channels using sectors 
    '''
    splitted  = [[None] * len(sectors) for i in range(len(channels_data))]

    for i_sector, sector in enumerate(sectors):
        a, b  = sector

        for i_channel, channel_data in enumerate(channels_data):
            splitted[i_channel][i_sector] = channel_data[a:b]

    return splitted

def get_person_segments(efd, target_channels):
    '''
    Get Subject's EDF segments from data by target_channels
    '''
    channels_data = subselect_channels(efd, target_channels)
    sectors, labels = extract_strict_sectors(efd)

    segments = [[None] * len(sectors) for i in range(len(channels_data))]

    for i_sector, sector in enumerate(sectors):
        a, b  = sector

        for i_channel, channel_data in enumerate(channels_data):
            segments[i_channel][i_sector] = channel_data[a:b]

    return segments, labels


# for binary classification. We select segments corresponding to the two selected marks
def get_two_segments_by_labels(segments, labels, label0 = 1, label1 = 2):
    segments_by_label0 = [[] for i in range(len(segments))]
    segments_by_label1 = [[] for i in range(len(segments))]

    for i_channel, channel in enumerate(segments):
        for i_segment, (segment, l) in enumerate(zip(channel, labels)):
            if l == label0:
                segments_by_label0[i_channel].append(segments[i_channel][i_segment])

            if l == label1:
                segments_by_label1[i_channel].append(segments[i_channel][i_segment])

    return (segments_by_label0, [0]*len(segments_by_label0[0])), (segments_by_label1, [1]*len(segments_by_label1[0]))

def normalize_labels(labels, min_phoneme_id = MIN_PHONEME_ID):
    return [p - min_phoneme_id for p in labels]