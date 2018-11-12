# tgb_recommender

运行前先将source/constants.py和server/serve/constants.py中的_ROOT值更改为存放代码的文件夹的路径，默认值为'usr/recommender'

运行实时更新脚本：	
python3.6 source/run.py [-c] [-l] 
如果使用可选参数-l，则先从本地文件读取先前已经获得的数据，再进行实时更新
如果使用可选参数-c, 则从配置文件中读取消息队列连接信息，否则使用默认值'localhost'

运行生成推荐脚本：
python3.6 server/manage.py runserver

HTTP请求的url: http://127.0.0.1:8000/serve/
