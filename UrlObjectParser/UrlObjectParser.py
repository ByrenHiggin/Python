import re

class UrlItem():
    def __init__(self,search, defaultProtocol = "http", defaultIsSecure = True, defaultFileExtension = "html"):
        self.QueryStrings = {}
        self.isSecure = True if search.group(2) == 's' else False    
        self.protocol = search.group(1) + 's' if self.isSecure is True else ''
        if self.protocol is None or self.protocol is "":
            self.protocol = defaultProtocol
        self.domain = search.group(3) if search.group(4) is None else search.group(4)
        self.subDomain = search.group(4) if not search.group(3) is None else ''
        self.domainExtension = search.group(5)
        self.path = search.group(6)
        self.fileName = search.group(7)
        self.GetQueryStringDict(search.group(8))
    def GetQueryStringDict(self, qrsl):
        arr = qrsl.replace("?","").split("&")
        for ele in arr:
            i = ele.split("=")
            self.QueryStrings[i[0]] = i[1]

class UrlObjectParser():
    def clean(self,url):
        search = re.match(r"(http?(s)?)?(?::\/\/)?(?:www.)?([a-zA-Z0-9]*)?(?:.)([a-zA-Z]{4,})?([.a-zA-Z]+){0,3}([\/a-zA-Z0-9]*(?:\/))*(?:\/)?([a-zA-Z0-9_.]+)([a-zA-Z0-9?&_=\-]*)",url)   
        if not search is None:
            Item = UrlItem(search,"https",True )        
