import json
import argparse
import requests
import yaml


def main(args):
    with open('../config/config.yml', 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(config['paths']['topics'], 'r') as f:
        topics = json.load(f)
    tid = args.topic_id
    print(topics[str(tid)]['body'])
    query_dict = {'topicID': str(tid)}
    if args.s:
        r = requests.get('http://127.0.0.1:8000/serve_special/', params=query_dict)
    else:
        r = requests.get('http://127.0.0.1:8000/serve/', params=query_dict)
    print('*'*80)
    print('您可能感兴趣的内容...')
    response = r.json()
    recoms = response['dto']['list']
    print(recoms)
    for tid, match_val in recoms:
        print('*'*80)
        print(topics[tid]['body'], match_val)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("topic_id", type=str, help="the topic for which recommendations are provided")
    parser.add_argument('-s', action='store_true', help='serve recommendations for specials')
    args = parser.parse_args()
    main(args)
