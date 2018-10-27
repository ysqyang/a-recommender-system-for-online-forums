import pika
import constants as const
import json
import configparser
import utils


with open(const._TOPIC_FILE, 'r') as f:
    topics = json.load(f)

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='new_topics')
channel.queue_declare(queue='update_topics')
channel.queue_declare(queue='active_topics')
channel.queue_bind(exchange='x', queue='new_topics', routing_key='new')
channel.queue_bind(exchange='x', queue='update_topics', routing_key='update')
channel.queue_bind(exchange='x', queue='active_topics', routing_key='new')
channel.queue_bind(exchange='x', queue='active_topics', routing_key='update')

new, update, active = [], [], []

def on_new_topics(ch, method, properties, body):
    '''
    new_topic = json.loads(body)
    #print(new_topic)
    topic_id = new_topic['topicid']
    topics[topic_id] = {k:v for k, v in new_topic.items() if k != 'topicid'}
    with open('topics_test', 'w') as f:
        json.dump(topics, f)
    '''
    new.append(body)
    print('new_topics: ', new)

def on_update_topics(ch, method, properties, body):
    '''
    active_topic = json.loads(body)
    #print(active_topic)
    topic_id = active_topic['topicid']
    for attr in active_topic:
        if attr != 'topicid':
            topics[topic_id][attr] += active_topic[attr]

    with open('topics_test', 'w') as f:
        json.dump(topics, f)
    '''
    update.append(body)
    print('update_topics: ', update)

def on_active_topics(ch, method, properties, body):
    '''
    active_topic = json.loads(body)
    #print(active_topic)
    topic_id = active_topic['topicid']
    for attr in active_topic:
        if attr != 'topicid':
            topics[topic_id][attr] += active_topic[attr]

    with open('topics_test', 'w') as f:
        json.dump(topics, f)
    '''
    active.append(body)
    print('active topics: ', active)

channel.basic_consume(on_new_topics,
                      queue='new_topics',
                      no_ack=True)

channel.basic_consume(on_update_topics,
                      queue='update_topics',
                      no_ack=True)

'''
channel.basic_consume(on_active_topics,
                      queue='active_topics',
                      no_ack=True)
'''

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()