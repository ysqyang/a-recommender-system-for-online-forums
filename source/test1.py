import pika
import constants as const
import json
import configparser
import utils


text = '"\\"当陈都灵告别了清纯，连体裤配蕾丝又美又帅，网友：糟糕我恋爱了 时 尚娱美人儿 2018-09-25 11:39:10《左耳》是饶雪漫的一部小说，讲述了李珥、张漾、许弋、黎吧啦等一群性格不同的年轻人在青春时期的疼痛的故事，后来小说被苏有朋改编成了[tag]电影[/tag]，而电影女主的李珥的扮演者就是陈都灵。这部电影是陈都灵的第一部作品 ， 也是她第一次担任女主的作品。虽然陈都灵出道已经有4年的时间了，但是也许是因为她的志向并不在演艺圈，所以她的作品并不是很多，偶尔会在影视剧里面客串角色，但是她唯一让人记住的角色只有《左耳》的“李珥”。陈都灵的长相清新，穿搭也是清新靓丽的样子，当她转换风格的时候又是别有一番滋味。；栏；陈都灵一向以清纯的气质示人，连平时的搭配都十分的清新自然，但是当陈都灵告别了清纯的时候，造型又是另外的风格了。她身穿一件黑色的连体裤搭配白色的蕾丝，看着既性感又帅气。 陈都灵这一次的穿搭风格完全改变 了，连体裤配蕾丝又美又帅，穿的是一件黑色的连体裤显得十分的帅气，而搭配白色的蕾丝内衬又带着女性的柔美，让陈都灵的造型不会过于强势，时尚感满满的。&nbsp;&nbsp;陈都灵虽然凭借《左耳》这部电影获得了很多的关注， 但是她一直都很低调，没想到陈都灵还 这么会穿搭；她这一身的造型是又酷又美，很多网友看了都纷纷表示：糟糕我恋爱了！ 再 来看看陈都灵的其他搭配，她身穿一件条纹针织衫，搭配的是绿色的半身裙，这身穿搭看着既显得时尚好看，又突显了陈都灵清新靓丽的气质，其实穿搭还是很适合陈都灵的气质的！ 陈都灵身穿一件很@灵机道人\u200b@台灯像是运动服的上衣，搭配的是一条粉色的半身裙 ，整体看起来陈都灵的造型粉粉嫩嫩的，突显了陈都灵的少女感，只是妆容精致的陈都灵还带着一丝妩媚的气息！你喜欢陈都灵的哪身搭配？ 当陈都灵告别了清纯，连体裤配蕾丝又 美又帅，网友：糟糕我恋爱了 时尚娱美人儿 2018-09-25 11:39:10《左耳》是饶雪漫的一 部小说，讲述了李珥、张漾、许弋、黎吧啦等一群性格不同的年轻人在青春时期的疼痛的故事，后来小说被苏有朋改编成了电影，而电影女主的李珥的扮演者就是陈都灵。这部电影是陈都灵的第一部作品， 也是她第一次担任女主的作品。虽然陈都灵出道已经有4年的时间了，但是也许是因为她的志向并不在演艺圈，所以她的作品并不是很多，偶尔会在影视剧里面客串角色，但是她唯一让人记住的角色只有《左耳》的“李珥”。陈都灵的长相清新，穿搭也是清新靓丽的样子，当她转换风格的时候又是别有一番滋味。 陈都灵一向以清纯的气质示 人，连平时的搭配都十分的清新自然，但是当陈都灵告别了清纯的时候，造型又是另外的风格了。她身穿一件黑色的连体裤搭配白色的蕾丝，看着既性感又帅气。 陈都灵这一次的穿 搭风格完全改变了，连体裤配蕾丝又美又帅，穿的是一件黑色的连体裤显得十分的帅气，而搭配白色的蕾丝内衬又带着女性的柔美，让陈都灵的造型不会过于强势，时尚感满满的。&nbsp;&nbsp;陈都灵虽然凭借《左耳》这部电影获得了很多的关注， 但是她一直都很低调， 没想到陈都灵还这么会穿搭；她这一身的造型是又酷又美，很多网友看了都纷纷表示：糟糕我恋爱了！ 再来看看陈都灵的其他搭配，她身穿一件条纹针织衫，搭配的是绿色的半身裙 ，这身穿搭看着既显得时尚好看，又突显了陈都灵清新靓丽的气质，其实穿搭还是很适合陈都灵的气质的！ 陈都灵身穿一件很像是运动服的上衣，搭配的是一条粉色的半身裙，整体 看起来陈都灵的造型粉粉嫩嫩的，突显了陈都灵的少女感只是妆容精致的陈都灵还带着一丝妩媚的气息！你喜欢陈都灵的哪身搭配？\\"&nbsp;&nbsp;给哥哥哥哥哥哥&nbsp;"'

stopwords = utils.load_stopwords('/home/ysqyang/Projects/recommender-system-for-online-forums/stopwords.txt')

print(utils.preprocess(text, stopwords, 1/20, 1/2, 5, 2))