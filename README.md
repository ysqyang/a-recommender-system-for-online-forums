# tgb_recommender

运行前先将source/constants.py和server/serve/constants.py中的_ROOT值更改为存放代码的文件夹的路径，默认值为'usr/recommender'

运行实时更新脚本：	
python3.6 source/run.py [-l]
如果加上可选参数-l，则先从本地文件读取先前已经获得的数据，再进行实时更新

运行生成推荐脚本：
python3.6 server/manage.py runserver

HTTP请求的url: http://127.0.0.1:8000/serve/
