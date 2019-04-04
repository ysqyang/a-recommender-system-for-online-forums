# tgb_recommender
运行路径： recommender/source

运行实时更新脚本：  	
python3.6 run.py [-c] [-l]  
如果使用可选参数-l，则先从本地文件读取先前已经获得的数据，再进行实时更新  
如果使用可选参数-c, 则从配置文件中读取消息队列连接信息，否则使用默认值'localhost'. 

运行生成推荐脚本：  
python3.6 server/manage.py runserver. 

HTTP请求的url: http://127.0.0.1:8000/serve/. 
