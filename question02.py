import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--date', help='date of event', type=str)
parser.add_argument('-t', '--time', help='time of event', type=str)
args = parser.parse_args()

print(f'Event was on {args.date} at {args.time}')