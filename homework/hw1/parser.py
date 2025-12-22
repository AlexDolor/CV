import argparse

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<50}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='lk', help='optical flow method. Values: "lk" (for sparce), "farneback" (for dence)')
    parser.add_argument('--video_in', type=str, default='homework/hw1/data/video_painting.mp4', help='video to stabilize')
    parser.add_argument('--video_out_dir', type=str, default='homework/hw1/output/', help='directory for the output videos')
    parser.add_argument('--visualization_dir', type=str, default='homework/hw1/output/vis/', help='output directory for the visualizations')
    parser.add_argument('--smoothing_radius', type=int, default=30, help='smoothing_radius (in frames)')
    return parser