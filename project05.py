"""CarND Project 05 - Vehicle Detection and Tracking"""

from detection import *
from training import *
from tracking import *

def process_image(image, params):
    config, clf = params['clf_config'], params['clf']

    if 'windows' not in params:
        print('Creating windows ...')
        pyramid = [((64, 64),  [400, 500]),
                   ((96, 96),  [400, 500]),
                   ((128, 128),[450, 578]),
                   ((192, 192),[450, None]),
              ]

        image_size = image.shape[:2]
        params['windows'] = create_windows(pyramid, image_size)

    all_windows = params['windows']
    if params['cache_enabled']:
        cache = process_image.cache
        if cache['heatmaps'] is None:
            cache['heatmaps'] = collections.deque(maxlen=params['heatmap_cache_length'])

        if 'tracker' not in cache:
            cache['tracker'] = VehicleTracker(image.shape)
        frame_ctr = cache['frame_ctr']
        tracker = cache['tracker']
        cache['frame_ctr'] += 1

    windows = itertools.chain(*all_windows)

    measurements = multiscale_detect(image, clf, config, windows)
    if not params['cache_enabled']:
        current_heatmap = update_heatmap(measurements, image.shape)
        thresh_heatmap = current_heatmap
        thresh_heatmap[thresh_heatmap < params['heatmap_threshold']] = 0
        cv2.GaussianBlur(thresh_heatmap, (31,31), 0, dst=thresh_heatmap)

        labels = label(thresh_heatmap)
        im2 = draw_labeled_bboxes(np.copy(image), labels)

    else:
        Z = []
        current_heatmap = update_heatmap(measurements, image.shape)
        cache['heatmaps'].append(current_heatmap)
        thresh_heatmap = sum(cache['heatmaps'])

        thresh_heatmap[thresh_heatmap < params['heatmap_threshold']] = 0
        cv2.GaussianBlur(thresh_heatmap, (31,31), 0, dst=thresh_heatmap)
        labels = label(thresh_heatmap)

        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            Z.append((np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)))
        tracker.detect(Z)
        im2 = tracker.draw_bboxes(np.copy(image))

    return im2

def clear_cache():
    process_image.cache = {
        'meas': None,
        'heatmaps': None,
        'frame_ctr': 0
    }


from docopt import docopt

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Advanced Lane Lines 1.0')

    clear_cache()
    model_file = arguments['MODEL']

    print('Loading model ...')


    in_file = arguments['INPUT']
    out_file = arguments['OUTPUT']



    clear_cache()
    params = {}
    params['clf_config'] = config
    params['clf'] = clf
    params['windows'] = windows
    params['cache_enabled'] = True
    params['heatmap_cache_length'] = 25
    params['heatmap_threshold'] = 5

    print('Processing video ...')
    clip2 = VideoFileClip(in_file)
    vid_clip = clip2.fl_image(process_image)
    vid_clip.write_videofile(out_file, audio=False)
