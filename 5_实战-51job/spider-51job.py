import urllib.request
from urllib import parse


kw = input('请输入你要搜索的岗位关键词')
keyword = parse.quote(parse.quote(kw))
pageNum = 1

def main():
    #学习如何灵活控制关键符
    url = "https://search.51job.com/list/180000,000000,0000,00,9,99,"+keyword+",2,"+str(pageNum)+".html"
    savepath = "51job-python信息"

    #测试url的要求
    print(url)
    #测试请求函数
    # html = askURL(url)
    # print(html)

    # datalist = getData(html)
    # saveData(datalist,savepath)

#获取数据 发送请求
def askURL(url):
    #header中键值不能有丝毫更改
    header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36 Edg/91.0.864.71"}
    request = urllib.request.Request(url,headers=header)
    html = ''
    try:
        responde = urllib.request.urlopen(request)
        #我们的编码方式还是会改变的  不一定都是utf-8
        html = responde.read().decode('gbk')
    except Exception as e:
        if hasattr(e,"code"):
            print(e.code)
        if hasattr(e,"reason"):
            print(e.reason)
    return html

def getLink():
    pass


if __name__ == '__main__':
    main()