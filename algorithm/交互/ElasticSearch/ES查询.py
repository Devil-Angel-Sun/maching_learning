#导入包
from elasticsearch import Elasticsearch
from elasticsearch import helpers

#初始变量
http = '192.168.50.100:9200'
indexName = 'dc_est_data'
es = Elasticsearch(http)

#函数
## 1.列出所有不同对象
def getAllObj(indexName,objectName,size=30000):
    body = {
        "size":0,
        "aggs":{
            "result":{
                "terms":{
                    "field":objectName ,
                    "size":size,
                        }
                }
               }
            }
    result = es.search(index=indexName,body=body)
    result = [x['key'] for x in result['aggregations']['result']['buckets']]
    return result

## 2.单条件查询
body = {
    "size":30000,
    "query":{
        "term":{
            "name":"python"
        }
    }
}
# 查询name="python"的所有数据
es.search(index=indexName,body=body)

body = {
    "query":{
        "terms":{
            "name":[
                "python","android"
            ]
        }
    }
}
# 搜索出name="python"或name="android"的所有数据
es.search(index=indexName,body=body)

## 3.scroll，查询大量数据适用
batchsize = 100 #每一批数据量大小
queryData = es.search(index=indexName, scroll='5m', timeout='3s', size=batchsize)
mdata = queryData.get("hits").get("hits")
if not mdata:
    print 'empty!'

scroll_id = queryData["_scroll_id"]
total = queryData["hits"]["total"]
for i in range(total/batchsize):
    res = es.scroll(scroll_id=scroll_id, scroll='5m') #scroll参数必须指定否则会报错
    mdata += res["hits"]["hits"]
    
## 4.批量导入
action = [{
        "_index": "s2",
        "_type": "doc",
        "_source": {
            "title": i
        }
    } for i in range(10000000)]
helpers.bulk(es, action)
