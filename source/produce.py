from datetime import datetime
import json
import pika
import time
import yaml


def main():
    with open('config/config.yml', 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    path_cfg = config['paths']
    mq_cfg = config['message_queue']
    misc_cfg = config['miscellaneous']

    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.exchange_declare(exchange=mq_cfg['exchange_name'], exchange_type='direct')
    channel.queue_declare(queue='new_topics')
    channel.queue_declare(queue='special_topics')
    channel.queue_declare(queue='delete_topics')
    channel.queue_declare(queue='old_topics')

    with open(path_cfg['topics'], 'r') as f:
        topics = json.load(f)

    for tid, info in topics.items():
        t = datetime.strptime(info['POSTDATE'], misc_cfg['datetime_format'])
        topics[tid] = {'postDate': time.mktime(t.timetuple())*1000,
                       'body': info['body']}

    for tid in topics.keys():
        rec = topics[tid]
        rec['topicID'] = tid
        msg = json.dumps(rec)
        if tid in {'1506377', '1506414'}:
            channel.basic_publish(exchange=mq_cfg['exchange_name'],
                                  routing_key='special',
                                  body=msg)
        else:
            channel.basic_publish(exchange=mq_cfg['exchange_name'],
                                  routing_key='new',
                                  body=msg)

    msg = json.dumps({'topicID': '1506556'})
    channel.basic_publish(exchange=mq_cfg['exchange_name'],
                          routing_key='delete',
                          body=msg)

    connection.close()


if __name__ == '__main__':
    main()
