import json
import os
import argparse
import yaml


def main(args):
    with open('../config/config.yml', 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tid = args.topic_id

    path = config['paths']['special_topics'] if args.s else config['paths']['topics']

    with open(path, 'r') as f:
        topics = json.load(f)

    print(topics[tid])

    if args.s:
        path = os.path.join(config['paths']['special_save'], tid)
    else:
        path = os.path.join(config['paths']['topic_save'],
                            str(int(tid)//config['miscellaneous']['num_topic_files_per_folder']), tid)

    with open(path, 'r') as f:
        topic = json.load(f)

    print(topic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("topic_id", type=str, help="the topic for which recommendations are provided")
    parser.add_argument('-s', action='store_true', help='serve recommendations for specials')
    args = parser.parse_args()
    main(args)